import torch

####################
# ema helper
####################


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def forward_self_ensemble(model, lr, out_size):
    """
    forward use self ensemble strategy

    :param model: Lightning Module
    :type model: LightningModule
    :param lr: input
    :type lr: Tensor
    :param out_size: output size
    :type out_size: tuple
    """

    def _transform(v, op):
        v = v.float()
        v2np = v.data.cpu().numpy()
        if op == "v":
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == "h":
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == "t":
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.tensor(tfnp, dtype=torch.float, requires_grad=False, device="cuda")
        return ret

    lr_list = [lr]
    for tf in "v", "h", "t":
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = []
    for i, aug in enumerate(lr_list):
        if i > 3:
            _out_size = (out_size[1], out_size[0])
        else:
            _out_size = out_size
        sr = model.forward(aug, _out_size).detach()
        sr_list.append(sr)

    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], "t")
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], "h")
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], "v")

    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)
    return output
