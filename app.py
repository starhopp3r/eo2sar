import os
import argparse
from utils import *
from PIL import Image
from networks import *
from torchvision import transforms


def parse_args():
    desc = "Use U-GAT-IT to convert optical Earth Observation (EO) images to Synthetic Aperture Radar (SAR) images"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--model_path', type=str, default='results/eo2sar_params_latest.pt', help='Path to pre-trained eo2sar model')
    parser.add_argument('--img', type=str, default='eo.tif', help='Path to optical EO image')
    parser.add_argument('--out_img_size', type=int, default=256, help='Size of output image (px)')
    return parser.parse_args()


class UGATIT(object) :
    def __init__(self, args):
        self.ch = args.ch
        self.n_res = args.n_res
        self.n_dis = args.n_dis
        self.img_ch = args.img_ch
        self.device = args.device
        self.img_size = args.img_size
        self.model_path = args.model_path
        self.image_path = args.img
        self.img_name = os.path.splitext(os.path.basename(args.img))[0]
        self.out_img_size = args.out_img_size

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

    def build_model(self):
        """ DataLoader """
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.output_transform = transforms.Compose([
            transforms.Resize((self.out_img_size, self.out_img_size))
        ])

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

    def load(self, path):
        params = torch.load(path)
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        try:
            self.load(self.model_path)
            print("[*] eo2sar pre-trained model successfully loaded!")
        except FileNotFoundError:
            print("Model not found at the given path!")

        self.genA2B.eval(), self.genB2A.eval()

        img = Image.open(self.image_path)
        real_A = self.test_transform(img).unsqueeze(0)
        real_A = real_A.to(self.device)
        fake_A2B, _, _ = self.genA2B(real_A)
        A2B = RGB2BGR(tensor2numpy(self.output_transform(denorm(fake_A2B[0]))))
        cv2.imwrite("gen_sar_{}.png".format(self.img_name), A2B * 255.0)


if __name__ == '__main__':
    args = parse_args()
    if args is None:
      exit()

    gan = UGATIT(args)
    gan.build_model()
    gan.test()