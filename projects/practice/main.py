from torchreid.data import datamanager
import torchreid
import torch

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

    model = torchreid.models.build_model(
        name='resnet18',
        num_classes=data_manager.num_train_pids,
        loss='softmax',
        pretrained=False
    ).cuda()

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