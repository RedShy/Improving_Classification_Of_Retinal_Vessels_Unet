import torch
import torch.nn as nn

# encoding block
class encoding_block(nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """

    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True, dropout=False):
        super().__init__()

        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
            ]

        else:
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.PReLU(),
            ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(nn.Upsample(mode="bilinear", align_corners=True, scale_factor=2), nn.Conv2d(in_size, out_size, kernel_size=1))

        else:  # dont use this!
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.upsample(input1, output2.size()[2:], mode="bilinear", align_corners=True)

        return self.conv(torch.cat([output1, output2], 1))


class UNet(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, num_classes=1):
        super().__init__()

        start = 20

        # encoding
        self.conv1 = encoding_block(3, start * 4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = encoding_block(start * 4, start * 8)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = encoding_block(start * 8, start * 16)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = encoding_block(start * 16, start * 32)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # center
        self.center = encoding_block(start * 32, start * 64)

        # decoding
        self.decode4 = decoding_block(start * 64, start * 32)
        self.decode3 = decoding_block(start * 32, start * 16)
        self.decode2 = decoding_block(start * 16, start * 8)
        self.decode1 = decoding_block(start * 8, start * 4)

        # final
        self.final = nn.Conv2d(start * 4, num_classes, kernel_size=1)

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decode4(conv4, center)

        decode3 = self.decode3(conv3, decode4)

        decode2 = self.decode2(conv2, decode3)

        decode1 = self.decode1(conv1, decode2)

        # final
        final = nn.functional.upsample(self.final(decode1), input.size()[2:], mode="bilinear", align_corners=True)

        return final

