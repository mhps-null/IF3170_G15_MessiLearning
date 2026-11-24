"""Animate the accepted FISTA iterates over a one-feature L1 loss contour."""

from pathlib import Path
from shutil import which

import numpy as np


def _parameter_grid(trajectory):
    margin = max(np.ptp(trajectory) * 0.16, 0.08)
    return np.linspace(trajectory.min() - margin, trajectory.max() + margin, 51)


def _l1_loss_grid(feature, labels, l1_strength, intercepts, coefficients):
    """Evaluate the mean logistic loss and L1 penalty on a parameter grid."""
    label_sign = 1.0 - 2.0 * labels
    grid = np.empty((len(coefficients), len(intercepts)))
    for row, coefficient in enumerate(coefficients):
        logits = intercepts[:, None] + coefficient * feature
        losses = np.logaddexp(0.0, label_sign * logits)
        grid[row] = (losses / len(labels)).sum(axis=1) + l1_strength * abs(coefficient)
    return grid


def _save_animation(animation, output_path, fps):
    import matplotlib as mpl
    from matplotlib.animation import FFMpegWriter, PillowWriter

    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            # Use the GIF writer if neither FFmpeg installation is available.
            pass

    with mpl.rc_context({"animation.ffmpeg_path": ffmpeg} if ffmpeg else {}):
        if output_path.suffix != ".gif" and FFMpegWriter.isAvailable():
            output_path = output_path.with_suffix(".mp4")
            writer = FFMpegWriter(fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"])
        else:
            output_path = output_path.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
        animation.save(output_path, writer=writer, dpi=90)
    return output_path

DEFAULT_TRAIN_DATA = Path(__file__).resolve().parents[1] / "data/processed/train_clean.csv"

def generate_loss_contour_video(
    data=DEFAULT_TRAIN_DATA,
    feature_column="days_before_arrival",
    target_column="is_canceled",
    output_path="output/videos/logreg_loss_contour.mp4",
):
    """Fit and animate one numeric feature from a prepared DataFrame or CSV.

    By default, use the official hotel training partition already transformed
    by the team's training-fitted pipeline. Return the written video path.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.animation import FuncAnimation

    from .logistic_regression import make_booking_logistic

    columns = [feature_column, target_column]
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data, usecols=columns)

    labels, classes = pd.factorize(data[target_column], sort=True)
    if len(classes) != 2:
        raise ValueError("The contour demonstration requires two target labels.")
    feature_matrix = data[[feature_column]].to_numpy()
    model = make_booking_logistic(len(feature_matrix), record_trajectory=True)
    model.fit(feature_matrix, labels)

    feature = feature_matrix[:, 0].astype(float, copy=False)
    intercepts, coefficients, objectives = np.array([
        (intercept, weights[0], loss) for intercept, weights, loss in model.trajectory_
    ]).T
    frames = np.linspace(0, model.n_iter_, min(len(objectives), 120), dtype=int)
    intercept_grid = _parameter_grid(intercepts)
    coefficient_grid = _parameter_grid(coefficients)
    grid = _l1_loss_grid(feature, labels, model.reg_lambda, intercept_grid, coefficient_grid)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, surface = plt.subplots(figsize=(7, 5), layout="constrained")
    try:
        feature_label = feature_column.replace("_", " ")
        surface.set(
            xlabel=r"Intercept $\theta_0$",
            ylabel=rf"{feature_label.capitalize()} coefficient $\theta_1$",
        )
        colors = surface.contourf(intercept_grid, coefficient_grid, grid, levels=16, cmap="cividis")
        surface.contour(
            intercept_grid, coefficient_grid, grid, levels=colors.levels[1:-1],
            colors="#eeeeee", linewidths=0.55, alpha=0.6,
        )
        trajectory, = surface.plot([], [], color="#e96736", linewidth=2.0)
        current, = surface.plot([], [], "o", color="#e96736", markeredgecolor="white", markersize=8)

        def update(iteration):
            through = slice(0, iteration + 1)
            trajectory.set_data(intercepts[through], coefficients[through])
            current.set_data([intercepts[iteration]], [coefficients[iteration]])
            surface.set_title(
                f"Iteration {iteration:,} / {model.n_iter_:,}  |  "
                f"Objective {objectives[iteration]:.6f}"
            )

        update(frames[0])
        fig.canvas.draw()
        fig.set_layout_engine("none")
        fps = 12
        # Hold both endpoints for one second so their titles can be read.
        video_frames = np.concatenate((
            np.repeat(frames[0], fps), frames, np.repeat(frames[-1], fps)
        ))
        animation = FuncAnimation(
            fig, update, frames=video_frames, interval=1000 / fps, cache_frame_data=False
        )
        return _save_animation(animation, output_path, fps)
    finally:
        plt.close(fig)
