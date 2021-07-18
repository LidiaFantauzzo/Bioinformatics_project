from models import BiSeNetV2, UNet, DeepLabV3_ResNet18


def create_model(args):

    if args.model == 'bisenetv2':
        model = BiSeNetV2(args, output_aux=args.output_aux)
    elif args.model == 'deeplabv3':
        model = DeepLabV3_ResNet18(args)
    elif args.model == 'unet':
        model = UNet(args)
    else:
        raise NotImplementedError("Specify a correct head.")

    return model
