import torch as t
import torchvision
import pretrainedmodels

from typing import (
    Dict,
)

from torch import (
    nn,
    Tensor,
)
from efficientnet_pytorch import EfficientNet


__all__ = (
    'Identity',
    'Swish',
    'MemoryEfficientSwish',
    'ResNetBackbone',
    'Backbone',
    'EfficientNetBackbone',
    'get_backbone_by_name',
)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class Swish(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x * t.sigmoid(x)


class SwishImplementation(t.autograd.Function):

    # skip typing: un-compatible signiture
    @staticmethod
    def forward(ctx, x):
        result = x * t.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    # skip typing: un-compatible signiture
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid_x = t.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return SwishImplementation.apply(x)


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        self.f_dim: int = -1
        self.has_swish: bool = False

    def extract_features(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    # (b, C, H, W) -> (b, f_dim)
    # ex: (64, 3, 224, 224) -> (32, 2048)
    def forward(self, x: Tensor, freeze_cnn: bool = False) -> Tensor:
        if freeze_cnn:
            with t.no_grad():
                features = self.extract_features(x)
        else:
            features = self.extract_features(x)
        return features


class ResNetBackbone(Backbone):

    RESOLUTION: int = 224
    TIER_TO_FEATURE_DIM: Dict[str, int] = {
        '152': 2048,
        '101': 2048,
        '50': 2048,
        '34': 512,
        '18': 512,
    }

    def __init__(self, tier: str = '152'):
        super(ResNetBackbone, self).__init__()

        # parameters
        self.tier = tier
        if self.tier not in self.TIER_TO_FEATURE_DIM:
            raise ValueError(f"Available choices for `tier`: {tuple(self.TIER_TO_FEATURE_DIM.keys())}, but got {self.tier}.")
        self.f_dim = self.TIER_TO_FEATURE_DIM[self.tier]

        # networks
        self.cnn: torchvision.models.resnet.ResNet = pretrainedmodels.__dict__[f'resnet{self.tier}'](num_classes=1000, pretrained='imagenet')
        self.cnn.last_linear = Identity()

    def extract_features(self, x: Tensor) -> Tensor:
        return self.cnn(x)


class EfficientNetBackbone(Backbone):

    TIER_TO_RESOLUTION: Dict[str, int] = {
        'b0': 224,
        'b1': 240,
        'b2': 260,
        'b3': 300,
        'b4': 380,
        'b5': 456,
        'b6': 528,
        'b7': 600,
    }
    TIER_TO_FEATURE_DIM: Dict[str, int] = {
        'b0': 1280,
        'b1': 1280,
        'b2': 1408,
        'b3': 1536,
        'b4': 1792,
        'b5': 2048,
        'b6': 2304,
        'b7': 2560,
    }

    def __init__(self, tier: str = 'b0'):
        super(EfficientNetBackbone, self).__init__()

        # swish checker
        self.has_swish = True

        # parameters
        self.tier = tier
        if self.tier not in self.TIER_TO_FEATURE_DIM:
            raise ValueError(f"Available choices for `tier`: {tuple(self.TIER_TO_FEATURE_DIM.keys())}, but got {self.tier}.")
        self.f_dim = self.TIER_TO_FEATURE_DIM[self.tier]

        # networks
        self.cnn: EfficientNet = EfficientNet.from_pretrained(f'efficientnet-{self.tier}')

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.cnn.extract_features(x)
        x = self.cnn._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def set_swish(self, memory_efficient: bool) -> None:
        if memory_efficient:
            self.cnn._swish = MemoryEfficientSwish()
            for block in self.cnn._blocks:
                block._swish = MemoryEfficientSwish()
        else:
            self.cnn._swish = Swish()
            for block in self.cnn._blocks:
                block._swish = Swish()


def get_backbone_by_name(name: str) -> Backbone:
    prefix, tier = name.split('-')
    if prefix == 'ResNet':
        return ResNetBackbone(tier)
    elif prefix == 'EfficientNet':
        return EfficientNetBackbone(tier)
    raise NotImplementedError
