from torchreid.data import datamanager
import torchreid
import torch


class MyModel(torch.nn.Module):

    def __init__(self, num_classes: int, loss: str):
        super().__init__()
        self.num_classes = num_classes
        self.loss = loss
        self.extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5), torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Sequential(torch.nn.Linear(16, num_classes))

    def forward(self, x: torch.Tensor):
        v = self.extractor.forward(x)
        v = v.view(v.size(0), -1)

        y = self.classifier.forward(v)

        return torch.ones((x.shape[0], self.num_classes))

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v


if __name__ == '__main__':
    data_manager = torchreid.data.ImageDataManager(
        root='data',
        sources='grid',
        targets='grid',
        batch_size_train=64,
        workers=0,
        transforms=['random_flip', 'random_crop'],
        train_sampler='RandomIdentitySampler' # this is important
    )

    # model = torchreid.models.build_model(
    #     name='resnet18',
    #     num_classes=data_manager.num_train_pids,
    #     loss='softmax',
    #     pretrained=False
    # ).cuda()
    model = MyModel(data_manager.num_train_pids, loss='softmax').cuda()

    # torchreid.utils.load_pretrained_weights(
    #     model, './log/model/model.pth.tar-3'
    # )

    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)

    scheduler = torchreid.optim.build_lr_scheduler(optimizer, stepsize=20)

    engine = torchreid.engine.ImageSoftmaxEngine(
        data_manager, model, optimizer, scheduler=scheduler
    )

    # engine.run(max_epoch=1, print_freq=1)
    engine.run(test_only=True, visrank=True)

    print('done')