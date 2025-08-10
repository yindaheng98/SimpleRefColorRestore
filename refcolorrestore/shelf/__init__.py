from .arch import rt4ksr_rep, rt4krestore_rep
from .archnerf import rt4kdual_rep
from .srvgg_arch import srvgg, restorevgg, dualvgg
from .srresnet_arch import srresnet, restoreresnet, dualresnet
model_dict = {
    "rt4ksr": rt4ksr_rep,
    "rt4krestore": rt4krestore_rep,
    "rt4kdual": rt4kdual_rep,
    "srvgg": srvgg,
    "restoreresnet": restoreresnet,
    "dualresnet": dualresnet,
    "srresnet": srresnet,
    "restorevgg": restorevgg,
    "dualvgg": dualvgg
}
