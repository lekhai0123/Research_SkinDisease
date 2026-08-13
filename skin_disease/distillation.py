import torch.nn.functional as F


def kd_kl_loss(student_logits, teacher_logits, temperature=4.0):
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)
