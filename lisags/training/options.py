"""Training command line: parser construction and option-combination checks.

Checks here depend only on the command line. Checks that need the resolved
method geometry or a source checkpoint live in training.source.
"""

import argparse

from lisags.models import add_method_arguments


def build_parser(argv=None):
    """Return the train.py parser; ``argv`` selects the method's option group."""
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr-decay-steps", type=int,
                   help="Finish means/network/train-camera LR decay at this step, then hold it fixed; defaults to --steps")
    p.add_argument("--save-steps", type=int, nargs="*", default=[],
                   help="Explicit intermediate checkpoints, without evaluating or selecting them during training")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--points", type=int, default=20000)
    p.add_argument("--max-points", type=int, default=400000)
    add_method_arguments(p, argv)
    p.add_argument("--port-start", type=int, default=5000)
    p.add_argument("--shadow-start", type=int, default=5000)
    p.add_argument("--refine-start", type=int, default=500, help="First densification/pruning step")
    p.add_argument("--refine-stop", type=int, default=25000)
    p.add_argument("--grow-grad2d", type=float,
                   help="Densification gradient threshold; omitted keeps 0.0008 with absgrad, else 0.0002")
    p.add_argument("--split-scale2d-stop", type=int,
                   help="Last step that splits every Gaussian whose footprint exceeds 3%% of the image, "
                        "independent of its gradient; omitted keeps --refine-stop, 0 disables the rule")
    p.add_argument("--budget-ramp", type=int, default=0,
                   help="Raise the point cap linearly from the initial count at --refine-start to --max-points "
                        "at this step; 0 applies --max-points from the start")
    p.add_argument("--opacity-reset-every", type=int, default=3000,
                   help="Opacity reset interval before --refine-stop; 0 disables resets. Large-Gaussian pruning "
                        "keeps its fixed 3000-step start")
    p.add_argument("--validate-every", type=int, default=0,
                   help="Validation/checkpoint interval; 0 disables all validation, including the final one, "
                        "and writes only last.pt")
    p.add_argument("--val-limit", type=int, default=100000)
    p.add_argument("--background", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-checkpoint")
    p.add_argument('--initialization', choices=['camera', 'hull'], default='camera',
                   help='Fresh Gaussian seeds: camera cube or training-silhouette-supported volume')
    p.add_argument("--freeze-geometry", action="store_true")
    p.add_argument("--unit-light-intensity", type=float)
    p.add_argument("--mask-weight", type=float, default=0.05)
    p.add_argument("--foreground-appearance", action="store_true",
                   help="Weight the photometric gradient reaching foreground radiance by GT alpha, so pixels of "
                        "the GT background supervise coverage only and cannot drive the shared appearance "
                        "networks (e.g. into saturation while random initial Gaussians cover a black background)")
    p.add_argument("--foreground-appearance-until", type=int, default=0,
                   help="Apply the --foreground-appearance weighting only for steps <= N (early-collapse guard that "
                        "later lets fringe pixels be shaded toward the background); 0 adds nothing")
    p.add_argument("--highlight-weight", type=float, default=0.,
                   help="Extra RGB and local-contrast loss on training-only neutral bright peaks")
    p.add_argument("--init-radius", type=float)
    p.add_argument("--display-gamma", type=float, default=2.2)
    p.add_argument("--loss-domain", choices=["observation", "linear"], default="observation",
                   help="Photometric training domain; evaluation always retains --display-gamma")
    p.add_argument("--opacity-cap", type=float, default=0.99)
    p.add_argument("--min-scale", type=float, default=1e-4)
    p.add_argument("--max-scale", type=float, default=0.1)
    p.add_argument("--position-scale", choices=["camera", "object"], default="camera")
    p.add_argument("--densification-scale", choices=["camera", "object"], default="object")
    p.add_argument("--position-decay", type=float, default=0.01)
    p.add_argument("--optimize-cameras", action="store_true")
    p.add_argument("--camera-start", type=int, default=1000)
    p.add_argument("--camera-lr", type=float, default=0.0003)
    p.add_argument("--camera-mode", choices=["anchor", "rotation"], default="anchor",
                   help="anchor: bounded SE(3) with the first fit camera fixed; rotation: all fit cameras "
                        "rotate about their fixed calibrated centers")
    p.add_argument("--camera-lr-final", type=float,
                   help="Exponentially decay the camera learning rate from --camera-lr at --camera-start "
                        "to this value at the final step; omitted keeps it constant")
    p.add_argument("--camera-gauge", choices=["none", "translation"], default="none",
                   help="translation: after each camera step, remove the part of the rotation corrections that "
                        "is equivalent to translating the whole scene and translate the scene instead")
    p.add_argument("--optimize-lights", action="store_true",
                   help="Fit per-frame point-light position offsets on fit frames; held-out views keep calibration")
    p.add_argument("--light-start", type=int, default=10000)
    p.add_argument("--light-lr", type=float, default=0.001, help="Offset learning rate in object-radius units")
    p.add_argument("--light-lr-final", type=float, default=0.00001)
    p.add_argument("--optimize-light-scale", action="store_true",
                   help="Fit one positive scene-wide irradiance normalization from training images")
    p.add_argument("--shadow-mode", choices=["depth", "deep"], default="deep")
    p.add_argument("--absgrad", action=argparse.BooleanOptionalAction, default=True)
    warmup = p.add_argument_group("LiSA geometry warmup")
    warmup.add_argument("--geometry-warmup-steps", type=int, default=0,
                         help="First N steps use LiSA per-Gaussian light-normalized color; "
                              "neural relighting starts at N+1")
    warmup.add_argument('--radiometric-curriculum', action='store_true',
                         help='LiSA 3D geometry warmup: fit raw radiance to display targets first, transition '
                              'to linear targets over its final third, then retain linear-domain training')
    p.add_argument("--holdout-every", type=int, default=0,
                   help="Train-internal validation: hold out every Nth official train frame (offset N//2), an "
                        "interpolation split like the official test; 0 keeps whole light-angle groups (extrapolation)")
    p.add_argument(
        "--fit-all",
        action="store_true",
        help="Final fit on all official training frames; no validation-based checkpoint selection",
    )
    return p


def validate_arguments(parser, args):
    """Reject inconsistent option combinations and apply frozen-stage overrides."""
    if args.val_limit <= 0:
        parser.error('--val-limit must be positive')
    if args.radiometric_curriculum and (args.representation != 'light_atlas'
                                        or args.geometry_warmup_steps <= 0 or args.init_checkpoint):
        parser.error('Radiometric curriculum requires fresh LiSA 3D training with a geometry warmup')
    if args.initialization == 'hull' and (args.init_checkpoint or args.points < 4):
        parser.error('--initialization hull requires fresh training and at least four points')
    if args.opacity_reset_every < 0:
        parser.error('--opacity-reset-every must be nonnegative')
    if args.budget_ramp and args.budget_ramp <= args.refine_start:
        parser.error('--budget-ramp must lie after --refine-start')
    if args.grow_grad2d is not None and args.grow_grad2d <= 0:
        parser.error('--grow-grad2d must be positive')
    if args.foreground_appearance_until < 0:
        parser.error('--foreground-appearance-until must be nonnegative')
    if args.holdout_every and (args.holdout_every < 2 or args.fit_all or args.init_checkpoint):
        parser.error('--holdout-every needs N >= 2, a fresh run and no --fit-all')
    if args.camera_lr_final is not None and (not args.optimize_cameras or args.camera_lr_final <= 0):
        parser.error('--camera-lr-final requires --optimize-cameras and a positive value')
    if args.camera_gauge != 'none' and (not args.optimize_cameras or args.camera_mode != 'rotation'):
        parser.error('--camera-gauge requires --optimize-cameras with --camera-mode rotation')
    if args.optimize_lights and (args.light_lr <= 0 or args.light_lr_final <= 0 or args.light_start < 1):
        parser.error('Light offsets need positive rates/start and a trainable relighting stage')
    return args


def parse_arguments(argv=None):
    parser = build_parser(argv)
    return validate_arguments(parser, parser.parse_args(argv))
