import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
import cv2
import onnxruntime as ort

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except Exception:  # pragma: no cover - optional dependency
    FaceAnalysis = None
    face_align = None


class FaceFeatureExtractor(nn.Module):
    """
    使用冻结的 CLIP 图像编码器提取全局人脸外观/结构特征。
    输入: [-1, 1] 归一化的可见光人脸图像(N,3,H,W)
    输出: L2 归一化的特征向量(N, D)，用于外观一致性约束
    """

    def __init__(self, device, model_name="ViT-B/32"):
        super().__init__()
        model, _ = clip.load(model_name, device=device, jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.register_buffer(
            "mean", th.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", th.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x + 1.0) * 0.5
        x = th.clamp(x, 0.0, 1.0)
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        x = x.float()
        features = self.model.encode_image(x)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features


class ArcFaceFeatureExtractor(nn.Module):
    """
    使用 InsightFace ArcFace 识别模型提取身份特征（更适合识别/身份一致性）。
    输入: [-1, 1] 归一化的可见光人脸图像(N,3,H,W)
    输出: L2 归一化的特征向量(N, D)
    """

    def __init__(self, device, model_name="buffalo_l"):
        super().__init__()
        if FaceAnalysis is None or face_align is None:
            raise ImportError(
                "insightface 未安装或缺少 FaceAnalysis，无法使用 ArcFace 特征提取。"
            )
        ctx_id = 0 if device.type == "cuda" else -1
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            ort_providers = ["CPUExecutionProvider"]
        app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=ort_providers,
        )
        app.prepare(ctx_id=ctx_id)
        self.model = app
        self.recognition = app.models.get("recognition")
        if self.recognition is None:
            raise RuntimeError("ArcFace recognition 模型未加载成功。")

    def _embed_from_aligned(self, aligned):
        rec = self.recognition
        if hasattr(rec, "get_feat"):
            return rec.get_feat(aligned)
        if callable(rec):
            return rec(aligned)
        if hasattr(rec, "forward"):
            return rec.forward(aligned)
        raise RuntimeError("ArcFace recognition 模型不支持特征提取调用。")

    def forward(self, x):
        # ArcFace 期望 BGR, uint8, 112x112
        device = x.device
        x = (x + 1.0) * 0.5
        x = th.clamp(x, 0.0, 1.0)
        x = x[:, [2, 1, 0], :, :]  # RGB -> BGR
        x = (x * 255.0).clamp(0.0, 255.0)
        x = x.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy().astype(np.uint8)

        feats = []
        for i in range(x.shape[0]):
            faces = self.model.get(x[i])
            if faces:
                # 选面积最大的脸，直接用识别器给的 embedding
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                feat = face.embedding
                feat = np.asarray(feat).reshape(-1)
                feats.append(feat)
                continue

            # fallback: center-crop and resize when detection fails
            h, w = x[i].shape[:2]
            size = min(h, w)
            y0 = (h - size) // 2
            x0 = (w - size) // 2
            crop = x[i][y0:y0 + size, x0:x0 + size]
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
            try:
                feat = self._embed_from_aligned(aligned)
                feat = np.asarray(feat).reshape(-1)
            except Exception:
                return None
            feats.append(feat)
        if not feats:
            return None
        feat_dim = feats[0].shape[0]
        if any(f.shape[0] != feat_dim for f in feats):
            return None
        feats = np.stack(feats, axis=0)
        feats = th.from_numpy(feats).to(device)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats
