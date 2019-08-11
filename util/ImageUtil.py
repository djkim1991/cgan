import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


class ImageUtil:
    def __init__(self):
        self.num_img = 0

    # 생성된 MNIST 이미지를 10x10 Grid로 보여주는 plot 함수를 정의합니다.
    def save(self, samples):
        samples = samples.detach().cpu().numpy()
        if not os.path.exists('generated_output/'):
            os.makedirs('generated_output/')

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(10, 10)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            plt.imshow(sample.reshape(28, 28))

        plt.savefig('generated_output/%s.png' % str(self.num_img).zfill(3), bbox_inches='tight')
        self.num_img += 1
        plt.close(fig)
