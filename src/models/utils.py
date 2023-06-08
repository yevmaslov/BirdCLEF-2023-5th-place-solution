import torch


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def collate_dict(inputs):
    for k, v in inputs.items():
        if type(v) != torch.Tensor:
            collated_dict = collate(inputs[k])
            inputs[k] = collated_dict
    return inputs


def batch_to_device(batch, device):
    for k, v in batch.items():
        if type(v) == dict:
            for _k, _v in v.items():
                if len(v) == 1:
                    v = v[0].unsqueeze(0)
                v[_k] = _v.to(device)
            batch[k] = v

        else:
            if len(v) == 1:
                v = v[0].unsqueeze(0)
            batch[k] = v.to(device)
    return batch


def get_valid_steps(num_train_steps, n_evaluations):
    eval_steps = num_train_steps // n_evaluations
    eval_steps = [eval_steps * i for i in range(1, n_evaluations + 1)]
    return eval_steps
