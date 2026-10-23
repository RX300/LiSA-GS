"""One GPU per run, training-only light holdout, explicit checkpoints.

The loop order (sampling, corrections, rendering, loss terms, optimizer steps,
refinement and projection) is fixed; optional branches live in ``training/``.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from gsplat.strategy.ops import remove

from lisags.data.dataset import SceneDataset
from lisags.evaluation.evaluator import target_image, ssim, evaluate_samples, save_pair, render_observation
from lisags.models import resolve_config
from lisags.geometry.refinement import Refinement
from lisags.training.initialization import (select_frames, load_training_samples,
                                     scene_normalization, build_scene)
from lisags.training.options import parse_arguments
from lisags.training.schedule import learning_rates, set_training_stage, radiometric_target
from lisags.training.source import load_source, check_source_compatibility


def main():
    args = parse_arguments()
    decay_steps = args.steps if args.lr_decay_steps is None else args.lr_decay_steps
    if decay_steps <= 0 or any(step <= 0 or step >= args.steps for step in args.save_steps):
        raise ValueError("LR decay steps must be positive; saved steps must lie strictly inside training")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = resolve_config(vars(args))
    saved = load_source(args.init_checkpoint)
    config["sss_light_axes"] = "world"
    dataset = SceneDataset(
        args.scene, "train", args.resolution, unit_light_intensity=args.unit_light_intensity
    )
    assert dataset.metadata_path.name == "transforms_train.json"
    config["metadata_path"] = str(dataset.metadata_path)
    check_source_compatibility(args, config, saved)
    fit_indices, val_indices = select_frames(args, config, dataset, saved)
    sample_pool = fit_indices
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "split.json").write_text(
        json.dumps({"fit": fit_indices, "validation": val_indices}, indent=2) + "\n"
    )
    print(json.dumps({"event": "loading_train_only", "metadata_path": str(dataset.metadata_path),
                      "total": len(dataset), "fit": len(fit_indices), "validation": len(val_indices)}),
          flush=True)
    samples, peak_masks = load_training_samples(args, dataset, fit_indices)
    center, radius, light_scale = scene_normalization(args, samples, fit_indices)
    gaussians, transport, source_shift = build_scene(
        args, config, dataset, saved, output, fit_indices, val_indices, center, radius, light_scale, samples)
    camera_origins = torch.stack([samples[i]["c2w"][:3, 3] for i in fit_indices])
    camera_extent = 1.1 * (camera_origins - camera_origins.mean(0)).norm(dim=-1).max().item()
    position_scale = camera_extent if args.position_scale == "camera" else gaussians.radius
    density_scale = camera_extent if args.densification_scale == "camera" else gaussians.radius
    optimizers = gaussians.optimizers(position_scale)
    network_parameters = list(transport.parameters())
    network_optimizer = (torch.optim.Adam(network_parameters, lr=0.001, eps=1e-15)
                         if network_parameters else None)

    # Branch construction order is part of the RNG contract; keep it.
    light_scale_fit = None
    if args.optimize_light_scale:
        from lisags.training.pose import LightScaleFit
        light_scale_fit = LightScaleFit(transport)
    camera = None
    if args.optimize_cameras:
        from lisags.training.pose import CameraFit
        fixed_camera = args.freeze_geometry
        camera = CameraFit(args, config, saved, samples, fit_indices, gaussians, trainable=not fixed_camera)
    lights = None
    if args.optimize_lights:
        from lisags.training.pose import LightFit
        lights = LightFit(args, saved, fit_indices, gaussians)
    # Cumulative translation applied to the means by the camera gauge since calibration.
    scene_gauge_shift = (torch.zeros(3, device="cuda") if source_shift is None
                         else torch.tensor(source_shift, device="cuda", dtype=torch.float32))

    strategy = Refinement(
        refine_start_iter=args.refine_start,
        refine_stop_iter=args.refine_stop,
        refine_scale2d_stop_iter=args.refine_stop if args.split_scale2d_stop is None else args.split_scale2d_stop,
        grow_scale2d=0.03,
        # Split broad footprints; opacity and world size determine pruning.
        prune_scale2d=float("inf"),
        grow_grad2d=args.grow_grad2d if args.grow_grad2d is not None else 0.0008 if args.absgrad else 0.0002,
        reset_every=3000,
        opacity_reset_every=args.opacity_reset_every,
        absgrad=args.absgrad,
        max_points=args.max_points,
        budget_ramp=args.budget_ramp,
        initial_points=min(len(gaussians.params["means"]), args.max_points),
        # Native 3DGS densification uses the original projected-mean gradients.
        key_for_gradient="means2d",
    )
    if not args.freeze_geometry:
        strategy.check_sanity(gaussians.params, optimizers)
    state = strategy.initialize_state(density_scale)
    excess = len(gaussians.params["means"]) - args.max_points
    if excess > 0:
        if args.freeze_geometry:
            raise ValueError('A fixed Gaussian teacher requires max-points >= checkpoint point count')
        remove_mask = torch.zeros(len(gaussians.params["means"]), device="cuda", dtype=torch.bool)
        remove_mask[gaussians.params["opacities"].detach().argsort()[:excess]] = True
        remove(gaussians.params, optimizers, state, remove_mask)
    print(json.dumps({
        "event": "initialized",
        "center": gaussians.center.tolist(),
        "radius": gaussians.radius,
        "light_scale": transport.light_scale.item(),
        "points": len(gaussians.params["means"]),
        "opacity_max": gaussians.params["opacities"].sigmoid().max().item(),
        "min_scale_ratio": gaussians.params["scales"].min().exp().item() / gaussians.radius,
        "camera_extent": camera_extent,
        "position_scale": position_scale,
        "densification_scale": density_scale,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }), flush=True)
    validation = [samples[i] for i in val_indices[:: max(1, len(val_indices) // args.val_limit)][: args.val_limit]]
    history = (output / "history.jsonl").open("a", buffering=1)
    start = time.monotonic()
    best = -math.inf

    def checkpoint(step, path):
        torch.save(
            {
                "config": config,
                "step": step,
                "gaussians": gaussians.state_dict(),
                "transport": transport.state_dict(),
                "radius": gaussians.radius,
                "val_indices": val_indices,
                "fit_indices": fit_indices,
                "camera_offsets": None if camera is None else camera.offsets.state_dict(),
                "light_offsets": None if lights is None else lights.offsets.state_dict(),
                "scene_gauge_shift": scene_gauge_shift.tolist(),
            },
            path,
        )

    for step in range(1, args.steps + 1):
        geometry_only = set_training_stage(gaussians, transport, optimizers, step, args.geometry_warmup_steps)
        if args.geometry_warmup_steps and step == args.geometry_warmup_steps + 1:
            event = {"event": "relighting_start", "step": step, "warmup_rgb_reset": False}
            history.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
        sample_index = random.choice(sample_pool)
        sample = samples[sample_index]
        camera_active = camera is not None and camera.active(step, geometry_only)
        if camera_active:
            sample = camera.begin(step, sample, sample_index)
        light_active = lights is not None and lights.active(step, geometry_only)
        if light_active:
            sample = lights.begin(step, sample, sample_index)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if network_optimizer is not None:
            network_optimizer.zero_grad(set_to_none=True)
        if light_scale_fit is not None and not geometry_only:
            light_scale_fit.begin(transport)
        means_lr, network_lr = learning_rates(args, step, decay_steps, position_scale)
        optimizers["means"].param_groups[0]["lr"] = means_lr
        if network_optimizer is not None:
            network_optimizer.param_groups[0]["lr"] = network_lr
        predicted, alpha, info = render_observation(
            gaussians,
            transport,
            sample,
            args.background,
            step >= args.shadow_start,
            step >= args.port_start,
            1. if args.radiometric_curriculum or args.loss_domain == 'linear' else args.display_gamma,
            args.shadow_mode,
            absgrad=args.absgrad,
            geometry_only=geometry_only,
            # GT-background pixels supervise coverage only, never the shared appearance model.
            appearance_weight=(sample["alpha"] if (args.foreground_appearance or step <= args.foreground_appearance_until)
                               and sample["alpha"] is not None else None),
        )
        if args.radiometric_curriculum:
            target = radiometric_target(sample, args.background, args.display_gamma, step, args.geometry_warmup_steps)
        elif args.loss_domain == 'linear':
            target = radiometric_target(sample, args.background, args.display_gamma, 1, 1)
        else:
            target = target_image(sample, args.background, args.display_gamma)
        l1 = (predicted - target).abs().mean()
        loss_terms = {
            'l1': 0.8 * l1,
            'ssim': 0.2 * (1 - ssim(predicted.clamp(0, 1), target.clamp(0, 1))),
            'mask': l1.new_zeros(()),
            'feature_reg': l1.new_zeros(()) if geometry_only else 1e-05 * gaussians.params['features'].square().mean(),
            'camera_reg': l1.new_zeros(()),
        }
        if sample['alpha'] is not None:
            loss_terms['mask'] = args.mask_weight * (alpha - sample['alpha']).abs().mean()
        if args.highlight_weight:
            loss_terms['highlight'] = highlight_loss(predicted, target, peak_masks[sample_index], args.highlight_weight)
        if camera_active and camera.trainable:
            loss_terms["camera_reg"] = camera.regularization(sample_index)
        if light_active:
            loss_terms["light_reg"] = lights.regularization(sample_index)
        loss = sum(loss_terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {step}")
        if not args.freeze_geometry:
            strategy.step_pre_backward(gaussians.params, optimizers, state, step, info)
        loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()
        if not geometry_only and network_optimizer is not None:
            network_optimizer.step()
        if camera_active and camera.trainable:
            shift = camera.step(gaussians)
            if shift is not None:
                scene_gauge_shift += shift
        if light_active:
            lights.step()
        if light_scale_fit is not None and not geometry_only:
            light_scale_fit.step(transport)
        if not args.freeze_geometry:
            event = strategy.step_post_backward(gaussians.params, optimizers, state, step, info, packed=False)
            if event is not None:
                history.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
        if not args.freeze_geometry:
            gaussians.project_geometry(args.opacity_cap, args.min_scale, args.max_scale)

        if step % 100 == 0 or step == 1 or step == args.steps:
            row = {
                "step": step,
                "stage": ("geometry" if geometry_only else "relighting"),
                "frame_index": sample_index,
                "loss": loss.item(),
                "l1": l1.item(),
                "loss_terms": {name: value.item() for name, value in loss_terms.items()},
                "points": len(gaussians.params["means"]),
                "seconds": round(time.monotonic() - start, 2),
                "memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
            }
            if getattr(transport, "diagnostics", None):
                row["transport_stats"] = {name: value.item() for name, value in transport.diagnostics.items()}
            if light_scale_fit is not None:
                row.update(light_scale_fit.log_fields(transport))
            if camera is not None:
                row.update(camera.log_fields())
            if args.camera_gauge != 'none':
                row['scene_gauge_shift'] = scene_gauge_shift.tolist()
            if lights is not None:
                row.update(lights.log_fields())
            history.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step in args.save_steps:
            checkpoint(step, output / f"step_{step:06d}.pt")
        if step == args.steps or (args.validate_every > 0 and step % args.validate_every == 0):
            if not validation or args.validate_every == 0:
                checkpoint(step, output / "last.pt")
                continue
            render_options = dict(geometry_only=geometry_only)
            metrics, _ = evaluate_samples(
                gaussians,
                transport,
                validation,
                args.background,
                step >= args.shadow_start,
                step >= args.port_start,
                display_gamma=args.display_gamma,
                shadow_mode=args.shadow_mode,
                **render_options,
            )
            row = {"step": step, "validation": metrics, "seconds": round(time.monotonic() - start, 2)}
            if camera is not None:
                rotation_rms, translation_rms = camera.offsets.rms()
                row.update(camera_rotation_rms=rotation_rms, camera_translation_rms=translation_rms)
            history.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            checkpoint(step, output / "last.pt")
            if metrics["PSNR"] > best:
                best = metrics["PSNR"]
                checkpoint(step, output / "best.pt")
            val = validation[0]
            with torch.no_grad():
                preview, _, _ = render_observation(
                    gaussians, transport, val, args.background, step >= args.shadow_start,
                    step >= args.port_start, args.display_gamma, args.shadow_mode, **render_options)
                save_pair(output / "validation_pair.png", preview,
                          target_image(val, args.background, args.display_gamma))
    history.close()
    from lisags.utils.plot_loss import plot_loss

    loss_plot = plot_loss(output / "history.jsonl")
    print(json.dumps({"event": "complete", "loss_plot": str(loss_plot),
                      "best_validation_psnr": best if math.isfinite(best) else None,
                      "seconds": time.monotonic() - start}), flush=True)


def highlight_loss(predicted, target, mask, weight):
    """Weighted RGB and neutral local-contrast error on training-only GT neutral peaks."""
    count = mask.sum().clamp_min(1)

    def contrast(values):
        return values - torch.nn.functional.avg_pool2d(values[None, None], 11, 1, 5)[0, 0]
    error = (predicted - target).abs().mean(-1) + .5 * (contrast(predicted.amin(-1)) - contrast(target.amin(-1))).abs()
    return weight * (error * mask).sum() / count


if __name__ == "__main__":
    main()
