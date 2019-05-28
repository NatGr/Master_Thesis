"""file containing the losses used in the normal case and when applying knowledge distilation"""
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.backend import softmax


def categorical_crossentropy_from_logits(target, output):
    """usual categorical crossentropy loss using logits so that the models are compatible with
    knowledge_distillation_loss"""
    return categorical_crossentropy(target, output, from_logits=True)


def knowledge_distillation_loss(target, output, classical_loss_weight, temperature, num_classes):
    """ loss used for knowledge distillation

    :param target: one-hot encoded ground truth classes and teacher logits
    :param output: output of the student
    :param classical_loss_weight: weight (float between 0 and 1) of the classical cross entropy loss of the student
    :param temperature: temperature that was used for the teacher predictions
    :param num_classes: the number of classes used in this classification problem
    :return: the loss to backprop on for the student
    """
    target_true, target_teacher = target[:, :num_classes], target[:, num_classes:]
    output_normal, output_soft = softmax(output), softmax(output / temperature)

    return classical_loss_weight * categorical_crossentropy(target_true, output_normal) + \
        (1 - classical_loss_weight) * (temperature ** 2) * categorical_crossentropy(target_teacher, output_soft)
    # categorical_crossentropy when computed from outputs of softmax (default arguments) corresponds to
    # - sum target*log(output)
