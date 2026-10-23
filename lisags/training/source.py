"""Source-checkpoint loading and checks that need the resolved method geometry."""

import torch


def load_source(path):
    return torch.load(path, map_location="cuda", weights_only=False) if path else None


def check_source_compatibility(args, config, saved):
    """Load the final method state; frozen appearance inherits calibration and stage switches."""
    if args.geometry_warmup_steps and args.freeze_geometry:
        raise ValueError("Geometry warmup requires trainable geometry")
    if saved is None:
        return
    if saved["config"]["representation"] != args.representation:
        raise ValueError("--init-checkpoint requires the LiSA method")
    config['source_checkpoint_step'] = saved['step']
    config['training_steps_before'] = saved['config'].get('training_steps_before', 0) + saved['step']
    if (args.representation == 'light_atlas' and
            saved['config'].get('normal_model', 'covariance') != args.normal_model):
        raise ValueError('Changing LiSA normal parameter meanings requires fresh training')
    fixed_lisa = args.representation == 'light_atlas' and args.freeze_geometry
    if fixed_lisa:
        if saved['step'] <= saved['config'].get('geometry_warmup_steps', 0):
            raise ValueError('Frozen LiSA appearance requires completed geometry warmup')
        if saved.get('light_offsets') is not None or args.optimize_lights or args.optimize_light_scale:
            raise ValueError('Frozen LiSA appearance requires fixed source light calibration')
        for key in ['camera_mode', 'camera_gauge']:
            setattr(args, key, saved['config'][key])
            config[key] = saved['config'][key]
        args.camera_start = config['camera_start'] = 1
        args.camera_lr = config['camera_lr'] = 0.
        args.camera_lr_final = config['camera_lr_final'] = None
    if fixed_lisa:
        args.optimize_cameras = saved['camera_offsets'] is not None
        config['optimize_cameras'] = args.optimize_cameras
        for key in ['port_start', 'shadow_start']:
            value = 1 if saved['step'] >= saved['config'][key] else args.steps+1
            setattr(args, key, value)
            config[key] = value
