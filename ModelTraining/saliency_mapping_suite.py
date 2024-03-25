import matplotlib.pylab as plt
import numpy as np
import math
import shutil
import sys
import tensorflow as tf


################################### SAVE SELF ###################################
def save_script_backup():
    
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/OfflineTraining_BrainClick/saliency_mapping_suite.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/OfflineTraining_BrainClick/saliency_mapping_suite.py'

    # Saving.
    shutil.copyfile(original, target)
    
# Immediately saving script.   
save_script_backup()







def generate_path_inputs(baseline, input, alphas):
    # Expand dimensions for vectorized computation of interpolations.
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(input, axis=0)
    delta = input_x - baseline_x
    path_inputs = baseline_x + alphas_x * delta

    return path_inputs


def compute_gradients(model, path_inputs, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(path_inputs)
        predictions = model(path_inputs)
        outputs = predictions[:, target_class_idx]
        # Note: IG requires softmax probabilities; converting Inception V1 logits.
        # outputs = tf.nn.softmax(predictions, axis=-1)[:, target_class_idx]
    gradients = tape.gradient(outputs, path_inputs)

    return gradients


def generate_alphas(m_steps=50, method="riemann_trapezoidal"): #m_steps = 50
    m_steps_float = tf.cast(m_steps, float)  # cast to float for division operations.

    if method == "riemann_trapezoidal":
        alphas = tf.linspace(0.0, 1.0, m_steps + 1)  # needed to make m_steps intervals.
    elif method == "riemann_left":
        alphas = tf.linspace(0.0, 1.0 - (1.0 / m_steps_float), m_steps)
    elif method == "riemann_midpoint":
        alphas = tf.linspace(
            1.0 / (2.0 * m_steps_float), 1.0 - 1.0 / (2.0 * m_steps_float), m_steps
        )
    elif method == "riemann_right":
        alphas = tf.linspace(1.0 / m_steps_float, 1.0, m_steps)
    else:
        raise AssertionError("Provided Riemann approximation method is not valid.")

    return alphas


def integral_approximation(gradients, method="riemann_trapezoidal"):
    if method == "riemann_trapezoidal":
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    elif method == "riemann_left":
        grads = gradients
    elif method == "riemann_midpoint":
        grads = gradients
    elif method == "riemann_right":
        grads = gradients
    else:
        raise AssertionError("Provided Riemann approximation method is not valid.")

    # Average integration approximation.
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients


@tf.function
def integrated_gradients(
    model,
    baseline,
    input,
    target_class_idx,
    m_steps=50,
    method="riemann_trapezoidal",
    batch_size=32,
):

    # 1. Generate alphas.
    alphas = generate_alphas(m_steps=m_steps, method=method)

    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = generate_path_inputs(
            baseline=baseline, input=input, alphas=alpha_batch
        )

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(
            model=model,
            path_inputs=interpolated_path_input_batch,
            target_class_idx=target_class_idx,
        )

        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients, method=method)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (input - baseline) * avg_gradients

    return integrated_gradients


def convergence_check(model, attributions, baseline, input, target_class_idx):
    # Your model's prediction on the baseline tensor. Ideally, the baseline score
    # should be close to zero.
    baseline_prediction = model(tf.expand_dims(baseline, 0))
    baseline_score = tf.squeeze(baseline_prediction)[target_class_idx]
    # Your model's prediction and score on the input tensor.
    input_prediction = model(tf.expand_dims(input, 0))
    input_score = tf.squeeze(input_prediction)[target_class_idx]
    # Sum of your IG prediction attributions.
    ig_score = tf.math.reduce_sum(attributions)
    delta = ig_score - (input_score - baseline_score)
    try:
        # Test your IG score is <= 5% of the input minus baseline score.
        tf.debugging.assert_near(ig_score, (input_score - baseline_score), rtol=0.05)
        #tf.print("Approximation accuracy within 5%.", output_stream=sys.stdout)
    except tf.errors.InvalidArgumentError:
        tf.print(
            "Increase or decrease m_steps to increase approximation accuracy.",
            output_stream=sys.stdout,
        )

    # tf.print("Baseline score: {:.3f}".format(baseline_score))
    # tf.print("Input score: {:.3f}".format(input_score))
    # tf.print("IG score: {:.3f}".format(ig_score))
    # tf.print("Convergence delta: {:.3f}".format(delta))


def plot_img_attributions(model, baseline, img, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4): #  m_steps=50
    # Attributions
    ig_attributions = integrated_gradients(
        model=model,
        baseline=baseline,
        input=img,
        target_class_idx=target_class_idx,
        m_steps=m_steps,
    )

    convergence_check(model, ig_attributions, baseline, img, target_class_idx)

    attribution_mask = tf.math.abs(ig_attributions)

#     # Visualization
#     fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(13, 5))

#     axs[0, 0].set_title("Baseline Image")
#     axs[0, 0].imshow(baseline, cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
#     axs[0, 0].axis("off")

#     axs[0, 1].set_title("Original Image")
#     axs[0, 1].imshow(img, cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
#     axs[0, 1].axis("off")

#     axs[1, 0].set_title("IG Attribution Mask")
#     axs[1, 0].imshow(attribution_mask, cmap=cmap, aspect="auto")
#     axs[1, 0].axis("off")

#     axs[1, 1].set_title("Original + IG Attribution Mask Overlay")
#     axs[1, 1].imshow(attribution_mask, cmap=cmap, aspect="auto")
#     axs[1, 1].imshow(img, alpha=overlay_alpha, cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
#     axs[1, 1].axis("off")

#     plt.tight_layout()

    # return fig, attribution_mask
    return attribution_mask