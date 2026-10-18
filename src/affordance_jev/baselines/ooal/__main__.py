import argparse
import json

from ...common import read


def main():
    parser = argparse.ArgumentParser(description='OOAL on the unified UMD novel/manual benchmark')
    parser.add_argument('stage', choices=['prepare', 'train', 'infer'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--checkpoints')
    parser.add_argument('--checkpoint')
    parser.add_argument('--manifest', help='Optional inference manifest; never used for training')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--steps', type=int, help='Development smoke test only; omitted for the full configured run')
    args = parser.parse_args()
    config = read(args.config)
    if args.stage == 'prepare':
        from .data import prepare
        result = prepare(config)
    elif args.stage == 'train':
        if not args.output or not args.checkpoints:
            parser.error('train requires --output and --checkpoints')
        from .engine import train
        result = train(config, args.output, args.checkpoints, args.steps)
    else:
        if not args.output or not args.checkpoint:
            parser.error('infer requires --output and --checkpoint')
        from .engine import infer
        result = infer(config, args.checkpoint, args.output, args.manifest, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
