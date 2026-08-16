"""Plot the training/validation loss curve written by train_gpt.py.

    python scripts/plot_loss.py          # reads runs/log.txt, writes runs/loss.png
    python scripts/plot_loss.py --dark   # dark theme
    python scripts/plot_loss.py --log other.txt --out other.png

The log format is one record per line: "<step> <train|val> <loss>".
Safe to run while training is still going -- it just plots what exists so far.
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gpt2.log import DEFAULT_RUN_DIR, read_log as gpt2_read_log

# NOTE: the backend is chosen in main(), not here -- importing this module from a
# notebook must leave the inline backend alone or the figures never render.

# The val loss of OpenAI's released GPT-2 124M checkpoint measured on this same
# FineWeb-EDU val split. Comparable only because we use the same data and the same
# gpt2 BPE -- if you retokenize with a different vocab this number stops meaning
# anything and should come out of the plot.
GPT2_124M_BASELINE = 3.2924

# Slots 1 and 2 of a CVD-validated categorical palette. Blue/orange separate at
# dE 24.7 (light) / 26.8 (dark) under protanopia, so the two series stay distinct
# for colorblind readers; they are also direct-labelled, so hue is never the only
# channel carrying identity.
THEMES = {
    'light': dict(
        surface='#fcfcfb', page='#f9f9f7',
        ink='#0b0b0b', ink2='#52514e', muted='#898781',
        grid='#e1e0d9', axis='#c3c2b7',
        train='#2a78d6', val='#eb6834',
    ),
    'dark': dict(
        surface='#1a1a19', page='#0d0d0d',
        ink='#ffffff', ink2='#c3c2b7', muted='#898781',
        grid='#2c2c2a', axis='#383835',
        train='#3987e5', val='#d95926',
    ),
}


def read_log(path):
    """-> (train_steps, train_loss, val_steps, val_loss) as float arrays."""
    parsed = gpt2_read_log(path)      # parsing lives in the package, not here

    def unzip(rows):
        if not rows:
            return np.array([]), np.array([])
        a = np.array(rows, dtype=float)
        return a[:, 0], a[:, 1]

    return (*unzip(parsed['train']), *unzip(parsed['val']))


def smooth(y, window):
    """Centered rolling mean, with the window shrinking at the edges so the
    smoothed line spans the full x-range instead of floating off the ends."""
    if len(y) < 3:
        return y
    window = max(3, min(window, len(y)))
    kernel = np.ones(window)
    total = np.convolve(y, kernel, mode='same')
    count = np.convolve(np.ones_like(y), kernel, mode='same')
    return total / count


def plot(train_steps, train_loss, val_steps, val_loss, theme='light', out_path=None, total_steps=None):
    """Draw the curve and return the Figure. Saves to out_path if given; the caller
    owns closing it, so a notebook can just let the figure render."""
    c = THEMES[theme]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    fig.patch.set_facecolor(c['page'])
    ax.set_facecolor(c['surface'])

    # --- chrome: recessive, hairline, solid (dashes are reserved for the threshold) ---
    ax.grid(axis='y', color=c['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(c['axis'])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c['muted'], labelsize=9, length=0, pad=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('sans-serif')

    # --- the GPT-2 reference: a threshold, which is what dashing is for ---
    if GPT2_124M_BASELINE is not None:
        ax.axhline(GPT2_124M_BASELINE, color=c['muted'], linewidth=1.2,
                   linestyle=(0, (5, 4)), zorder=1)
        ax.text(0.995, GPT2_124M_BASELINE, f'  OpenAI GPT-2 (124M)  {GPT2_124M_BASELINE:.2f}',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=9, color=c['muted'])

    # --- train: raw per-step behind, smoothed in front (one entity, one hue) ---
    if len(train_steps):
        ax.plot(train_steps, train_loss, color=c['train'], linewidth=1.0,
                alpha=0.22, zorder=2, solid_capstyle='round')
        window = max(3, len(train_steps) // 60)
        ax.plot(train_steps, smooth(train_loss, window), color=c['train'],
                linewidth=2.0, zorder=4, solid_capstyle='round', label='Train')

    # --- val: sparse, so show the actual measurements ---
    if len(val_steps):
        ax.plot(val_steps, val_loss, color=c['val'], linewidth=2.0, zorder=5,
                marker='o', markersize=6, markerfacecolor=c['val'],
                markeredgecolor=c['surface'], markeredgewidth=1.5,
                solid_capstyle='round', label='Validation')

    # --- direct labels on the endpoints, so identity never rests on hue alone ---
    # train and val converge, so the two labels would sit on top of each other:
    # push the higher one up and the lower one down rather than both centered.
    ends = [(steps[-1], losses[-1], name)
            for steps, losses, name in ((train_steps, train_loss, 'Train'),
                                        (val_steps, val_loss, 'Validation'))
            if len(steps)]
    ends.sort(key=lambda e: e[1])
    offsets = {0: (8, 0)} if len(ends) == 1 else {0: (8, -9), 1: (8, 9)}
    for i, (x, y, name) in enumerate(ends):
        ax.annotate(f'{name}  {y:.2f}', xy=(x, y), xytext=offsets[i],
                    textcoords='offset points', va='center', ha='left',
                    fontsize=10, color=c['ink2'])

    last = int(max(train_steps[-1] if len(train_steps) else 0,
                   val_steps[-1] if len(val_steps) else 0))
    # room for the endpoint labels, but never dead space past a known finish line
    ax.set_xlim(0, max(total_steps or 0, last * 1.12))
    ax.set_xlabel('Step', fontsize=10, color=c['ink2'], labelpad=10)
    ax.set_ylabel('Cross-entropy loss', fontsize=10, color=c['ink2'], labelpad=10)

    ax.set_title('GPT-2 training loss', fontsize=15, color=c['ink'],
                 loc='left', pad=18, fontweight='semibold')
    subtitle = f'{last:,} steps'
    if total_steps:
        subtitle += f' of {total_steps:,}'
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=10, color=c['muted'])

    leg = ax.legend(loc='upper right', frameon=False, fontsize=10, handlelength=1.6)
    for text in leg.get_texts():
        text.set_color(c['ink2'])          # text wears ink, the swatch carries identity

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    return fig


def summarize(train_steps, train_loss, val_steps, val_loss):
    """The table view -- every plotted value stays reachable without reading pixels."""
    rows = []
    if len(train_loss):
        rows.append(('train  first', int(train_steps[0]), train_loss[0]))
        rows.append(('train  last', int(train_steps[-1]), train_loss[-1]))
        rows.append(('train  best', int(train_steps[np.argmin(train_loss)]), train_loss.min()))
    if len(val_loss):
        rows.append(('val    last', int(val_steps[-1]), val_loss[-1]))
        rows.append(('val    best', int(val_steps[np.argmin(val_loss)]), val_loss.min()))

    print(f"{'series':<14}{'step':>8}{'loss':>10}")
    print('-' * 32)
    for name, step, loss in rows:
        print(f'{name:<14}{step:>8,}{loss:>10.4f}')
    if len(val_loss):
        print('-' * 32)
        print(f'{"gap to GPT-2":<14}{"":>8}{val_loss.min() - GPT2_124M_BASELINE:>+10.4f}')


def main():
    matplotlib.use('Agg')   # script mode: write a file, never pop a window
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', default=os.path.join(DEFAULT_RUN_DIR, 'log.txt'))
    ap.add_argument('--out', default=None, help='default: <log dir>/loss.png')
    ap.add_argument('--dark', action='store_true')
    ap.add_argument('--both', action='store_true', help='write light and dark side by side')
    ap.add_argument('--show', action='store_true',
                    help='open the png in the default viewer after writing it')
    ap.add_argument('--total-steps', type=int, default=None,
                    help='draw the x-axis out to the full planned run')
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f'no log at {args.log} -- has training written anything yet?')

    data = read_log(args.log)
    if not len(data[0]) and not len(data[2]):
        raise SystemExit(f'{args.log} has no parseable records')

    out = args.out or os.path.join(os.path.dirname(args.log), 'loss.png')
    themes = ['light', 'dark'] if args.both else ['dark' if args.dark else 'light']
    written = []
    for theme in themes:
        path = out if len(themes) == 1 else out.replace('.png', f'_{theme}.png')
        fig = plot(*data, theme=theme, out_path=path, total_steps=args.total_steps)
        plt.close(fig)
        written.append(path)
        print(f'wrote {path}')

    if args.show:
        # Agg never opens a window -- hand the file to the OS viewer instead, which
        # also works over ssh-less remote shells and won't steal focus mid-training
        import subprocess, sys as _sys
        opener = {'darwin': ['open'], 'win32': ['cmd', '/c', 'start', '']}.get(_sys.platform, ['xdg-open'])
        for path in written:
            subprocess.run(opener + [path], check=False)

    print()
    summarize(*data)


if __name__ == '__main__':
    main()
