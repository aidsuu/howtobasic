from __future__ import annotations

import argparse
from pathlib import Path

from hydrogen3d import sample_orbital_points, make_3d_scatter_figure, save_html, save_samples_json


def validate_quantum_numbers(n: int, l: int, m: int) -> None:
    if n < 1:
        raise ValueError("n harus >= 1")
    if not (0 <= l <= n - 1):
        raise ValueError("l harus memenuhi 0 <= l <= n-1")
    if not (-l <= m <= l):
        raise ValueError("m harus memenuhi -l <= m <= l")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulasi orbital atom hidrogen 3D")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="data/samples")
    parser.add_argument("--complex", action="store_true", help="pakai orbital kompleks asli")
    parser.add_argument("--kind", type=str, default="c", choices=["c", "s"], help="jenis orbital real")
    args = parser.parse_args()

    validate_quantum_numbers(args.n, args.l, args.m)

    data = sample_orbital_points(
        n=args.n,
        l=args.l,
        m=args.m,
        n_samples=args.samples,
        seed=args.seed,
        real_orbital=not args.complex,
        real_kind=args.kind,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / f"orbital_n{args.n}_l{args.l}_m{args.m}.json"
    html_path = outdir / f"orbital_n{args.n}_l{args.l}_m{args.m}.html"

    save_samples_json(json_path, args.n, args.l, args.m, data)

    fig = make_3d_scatter_figure(
        data,
        title=f"Hydrogen orbital n={args.n}, l={args.l}, m={args.m}",
    )
    save_html(fig, html_path)

    print(f"Saved JSON : {json_path}")
    print(f"Saved HTML : {html_path}")


if __name__ == "__main__":
    main()