import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DualScaleFusionHead(nn.Module):
    def __init__(self, c4_channels, c5_channels, fusion_channels=512):
        super().__init__()
        self.c4_proj = nn.Sequential(
            nn.Conv2d(c4_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU(inplace=True)
        )
        self.c5_proj = nn.Sequential(
            nn.Conv2d(c5_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, c4, c5):
        c4 = self.c4_proj(c4)
        c5 = self.c5_proj(c5)
        c5 = F.interpolate(c5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([c4, c5], dim=1)
        x = self.fuse(x)
        return x


class EfficientNetB2DualScale(nn.Module):
    def __init__(self, num_classes, pretrained=True, drop_rate=0.0, fusion_channels=512):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.fusion_channels = fusion_channels

        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        self.num_stages = len(self.backbone.blocks)
        self.c4_stage_idx = self.num_stages - 3
        self.c5_stage_idx = self.num_stages - 1

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 260, 260)
            c4, c5 = self._infer_channels(dummy)

        c4_channels = c4.shape[1]
        c5_channels = c5.shape[1]

        self.fusion_head = DualScaleFusionHead(
            c4_channels=c4_channels,
            c5_channels=c5_channels,
            fusion_channels=fusion_channels
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )

        self.classifier = nn.Linear(fusion_channels, num_classes)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        if num_classes <= 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.fusion_channels, num_classes)

    def extract_dual_features(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)

        c4 = None
        c5 = None

        for i, stage in enumerate(self.backbone.blocks):
            x = stage(x)
            if i == self.c4_stage_idx:
                c4 = x
            if i == self.c5_stage_idx:
                c5 = x

        return c4, c5
    
    def _infer_channels(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)

        c4 = None
        c5 = None

        for i, stage in enumerate(self.backbone.blocks):
            x = stage(x)
            if i == self.c4_stage_idx:
                c4 = x
            if i == self.c5_stage_idx:
                c5 = x

        return c4, c5

    def forward_features(self, x):
        c4, c5 = self.extract_dual_features(x)
        x = self.fusion_head(c4, c5)
        return x

    def forward_head(self, x, pre_logits=False):
        x = self.pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def debug_shapes(self, x):
        print("input:", x.shape)

        x = self.backbone.conv_stem(x)
        print("after conv_stem:", x.shape)

        x = self.backbone.bn1(x)
        print("after bn1:", x.shape)

        c4 = None
        c5 = None

        for i, stage in enumerate(self.backbone.blocks):
            x = stage(x)
            print(f"after blocks[{i}]:", x.shape)
            if i == self.c4_stage_idx:
                c4 = x
                print(f"==> C4 captured at blocks[{i}]:", c4.shape)
            if i == self.c5_stage_idx:
                c5 = x
                print(f"==> C5 captured at blocks[{i}]:", c5.shape)

        x_head = self.backbone.conv_head(c5)
        print("after conv_head:", x_head.shape)

        x_head = self.backbone.bn2(x_head)
        print("after bn2:", x_head.shape)

        x_fused = self.fusion_head(c4, c5)
        print("after fusion_head:", x_fused.shape)

        x_pool = self.pool(x_fused)
        print("after pool:", x_pool.shape)

        x_out = self.classifier(x_pool)
        print("after classifier:", x_out.shape)

        return {
            "c4": c4.shape,
            "c5": c5.shape,
            "head": x_head.shape,
            "fused": x_fused.shape,
            "pooled": x_pool.shape,
            "logits": x_out.shape
        }

    def verify_feature_info(self):
        print("num_stages:", self.num_stages)
        print("c4_stage_idx:", self.c4_stage_idx)
        print("c5_stage_idx:", self.c5_stage_idx)
        print("feature_info:")
        for i, info in enumerate(self.backbone.feature_info):
            print(i, info)


class EfficientNetB2Original(nn.Module):
    def __init__(self, num_classes, pretrained=True, drop_rate=0.0):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )

    def get_classifier(self):
        return self.model.get_classifier()

    def reset_classifier(self, num_classes):
        self.model.reset_classifier(num_classes)

    def forward(self, x):
        return self.model(x)


def efficientnet_b2_original(num_classes, pretrained=True, drop_rate=0.0):
    return EfficientNetB2Original(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate
    )


def efficientnet_b2_dual_scale(num_classes, pretrained=True, drop_rate=0.0, fusion_channels=512):
    return EfficientNetB2DualScale(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        fusion_channels=fusion_channels
    )