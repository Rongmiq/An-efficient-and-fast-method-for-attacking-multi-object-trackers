import cv2
import os
import torch

# from CenterNet.src.lib.detectors.ctdet import CtdetDetector
# from CenterNet.src.lib.opts import opts as center_opts
# from CenterNet.src.lib.models.decode import *
# # # #
# from CenterTrack.src.lib.opts import opts as centertrack_opts
# from CenterTrack.src.lib.detector import Detector as centertrack_detector

# from FairMOT.src.lib.tracker.multitracker import JDETracker
# from FairMOT.src.lib.opts import opts as FairMOT_opts
#
# from TraDeS.src.lib.opts import opts as trades_opts
# from TraDeS.src.lib.detector import Detector as trades_detector

from ByteTrack.yolox.exp import get_exp
from ByteTrack.tools import track
from ByteTrack.yolox.evaluators import MOTEvaluator
from ByteTrack.yolox.utils import fuse_model
args = track.make_parser().parse_args()
exp = get_exp(args.exp_file, args.name)
exp.merge(args.opts)

class MOTTracker():
    #
    # def __init__(self):
    #     opt = center_opts().init()
    #     self.detector = CtdetDetector(opt)  # CenterNet Detect
    #
    # def get_maps(self, input, tracking=False):
    #     hm, wh, reg = self.detector.run(input, return_maps=True)
    #     #[B,80,128,256]
    #     return hm, reg, wh

    def __init__(self):
        opt = trades_opts().init()
        self.detector = trades_detector(opt)  # CtenerNet Tracking

    def get_maps(self, input, tracking=True):
        # input = input.to('cuda:1')
        hm, reg, wh, embedding, offset = self.detector.run(input, return_maps=True)
        # print(hm.shape) #[B,1,H/4, W/4]
        # print(reg.shape) #[B,2,H/4, W/4]
        # print(wh.shape) #[B,2,H/4, W/4]
        # print(embedding.shape) #[B,128,H/4, W/4]
        # print(offset.shape) #[B,2,H/4, W/4]
        # # print(hm) #[0-1]
        # print(reg)  #[-1,+1]
        # print(wh)  #[-1,100]
        # print(embedding) #[-1,+1]
        # print(offset)  #[0,1]
        # print(torch.max(offset),torch.min(offset))
        # assert p==0
    #
        if tracking:
            return hm, reg, wh, embedding, offset
        else:
            return hm, reg, wh


    # def __init__(self):
    #     opt = centertrack_opts().init()
    #     self.detector = centertrack_detector(opt)  # CenterNet Detect
    #
    # def get_maps(self, input):
    #     hm, wh, reg = self.detector.run(input, return_maps=True)
    #
    #     return hm, wh, reg

    def get_clean_results(self, input):
        hm,wh,reg = self.detector.run(input,return_maps=True)
        batch = hm.shape[0]
        K = 20
        hm_nms = self._nms(hm)
        topk_score, topk_inds, topk_clses, topk_ys, topk_xs = self._topk(hm_nms) #shape=[Batchsize,K]=[16,40]
        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, topk_inds)
            reg = reg.view(batch, K, 2)
            topk_xs = topk_xs.view(batch, K, 1) + reg[:, :, 0:1]
            topk_ys = topk_ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            topk_xs = topk_xs.view(batch, K, 1) + 0.5
            topk_ys = topk_ys.view(batch, K, 1) + 0.5

        wh = self._transpose_and_gather_feat(wh, topk_inds)
        return  topk_score, topk_inds, topk_clses,  topk_ys,   topk_xs,  reg,    wh
                 # [B,k]      [B,K ]      [B,K]     [B,K,1]    [B,K,1]   [B,K,2] [B,K,2]

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=20):
        batch, cat, height, width = scores.size()
        #[B,C,K]
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        #[B,K]
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def show_tensor(self, tensor):
        if len(tensor.shape) == 4 and tensor.shape[0] > 1:
            for i in range(tensor.shape[0]):
                img_np = tensor[i].squeeze(0).cpu().numpy()
                print(img_np.shape)
                for j in range(img_np.shape[0]):
                    cv2.imshow(str(j),img_np[j] * 255)
                    cv2.waitKey(50)
                    cv2.destroyAllWindows()
        else:
            img_np = tensor.squeeze(0).cpu().numpy()
            for j in range(img_np.shape[0]):
                cv2.imshow(str(j), img_np[j] * 255)
                cv2.waitKey(50)
                cv2.destroyAllWindows()

class FairMOT():
    def __init__(self):

        opt = FairMOT_opts().init()
        self.tracker = JDETracker(opt, frame_rate=30)

    def get_clean_maps(self, input):
        hm, wh, _, reg = self.tracker.update_for_train_generator_clean(input)
        return hm, wh, reg

    def get_adv_maps(self, input, id=False):
        hm, wh, id_feature, reg = self.tracker.update_for_train_generator_adv(input, id=id)
        if id:
            hm = self._nms(hm)
            scores, inds, clses, ys, xs = self._topk(hm, K=100)
            return inds, id_feature
        else:
            return hm, wh, id_feature, reg

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat


class ByteTrack():
    def __init__(self, exp=exp, args=args):
        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        model = exp.get_model()
        evaluator = MOTEvaluator(
            args=args,
            dataloader = None,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
        )
        model.cuda()
        model.eval()
        if not args.speed and not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
            else:
                ckpt_file = args.ckpt
            print("loading checkpoint from", ckpt_file)
            loc = "cuda:0"
            ckpt = torch.load(ckpt_file, map_location=loc)
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            print("loaded checkpoint done.")

        self.evaluator = evaluator
        self.model = model
        self.fp16 = args.fp16
        self.test_size = exp.test_size

    def get_maps(self, imgs=None):
        outputs = self.evaluator.train_generator_with_yolox(self.model, imgs, self.fp16, self.test_size, )

        return outputs

if __name__ == '__main__':
    img = torch.ones(12,3,608,1088).cuda()
    FairMOT = FairMOT()
    for i in range(100):
        f = FairMOT.get_adv_maps(img)
        print(f)