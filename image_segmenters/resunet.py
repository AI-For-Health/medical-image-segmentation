# import torch
# import torch.nn as nn

# # train resunet on kvasir-seg dataset
# class ResUNet(nn.Module):
#     #define a resunet encoder block
#     def _block(self, in_channels, out_channels, kernel_size, stride=2, padding):
#         return nn.Sequential(
#             #batch normalization
#             nn.BatchNorm2d(in_channels),
#             # relu inplace
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
#             #batch normalization
#             nn.BatchNorm2d(out_channels),
#             # relu inplace
#             nn.ReLU(inplace=True),
#             # convolution
#             nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1),
#             # batch normalization
#             nn.BatchNorm2d(out_channels),
#             # relu inplace
#             nn.ReLU(inplace=True),
#         )

#     def _skip(self, in_channels, out_channels, kernel_size, stride, padding="valid"):
#         return nn.Sequential(
#             # convolution
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
#             # batch normalization
#             nn.BatchNorm2d(out_channels),
#         )
#     # _init_ method
#     def __init__(self, in_channels=3, out_channels=1, kernel_size=3):
#         super().__init__()
#         # level 1
#         self.level1=nn.Sequential(
#             # convolution
#             nn.Conv2d(in_channels, 64, kernel_size, stride=1, padding="same"),
#             # batch normalization
#             nn.BatchNorm2d(64),
#             # relu inplace
#             nn.ReLU(inplace=True),
#             # convolution
#             nn.Conv2d(64, 64, kernel_size, stride=1, padding="same"),
#             # batch normalization
#             nn.BatchNorm2d(64),
#             # relu inplace
#             nn.ReLU(inplace=True),
#         )
#         self.skip1=self._skip(in_channels, 64, kernel_size, stride=1, padding="same")
#         # level 2
#         self.level2=self._block(64, 128, kernel_size)
#         self.skip2=self._skip(64, 128, kernel_size, stride=2)
#         # level 3
#         self.level3=self._block(128, 256, kernel_size)
#         self.skip3=self._skip(128, 256, kernel_size, stride=2)
#         # level 4
#         self.level4=self._block(256, 512, kernel_size)
#         self.skip4=self._skip(256, 512, kernel_size, stride=2)
#         # upsample1
#         self.upsample1=nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         # level 5
#         self.level5=self._block(512, 256, kernel_size, stride=1)
#         self.skip5=self._skip(512, 256, kernel_size, stride=2)
#         # upsample2
#         self.upsample2=nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         # level 6
#         self.level6=self._block(256, 128, kernel_size, stride=1)
#         self.skip6=self._skip(256, 128, kernel_size, stride=2)
#         # upsample3
#         self.upsample3=nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         # level 7
#         self.level7=self._block(128, 64, kernel_size, stride=1)
#         self.skip7=self._skip(128, 64, kernel_size, stride=2)
#         # final convolution
#         self.final_conv=nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding="same")
#         # sigmoid activation
#         self.sigmoid=nn.Sigmoid()
    
#     # forward method
#     def forward(self, x):
#         # level 1
#         print("x: ", x.shape)
#         level1=self.level1(x)
#         print("level1: ", level1.shape)
#         # residual connection
#         level1 = level1 + self.skip1(x)
#         # level 2
#         level2=self.level2(level1)
#         print("level2: ", level2.shape)
#         # print(level1.shape)
#         # print(level2.shape)
#         # residual connection
#         level2 = level2 + self.skip2(level1)
#         # level 3
#         level3=self.level3(level2)
#         # residual connection
#         level3 = level3 + self.skip3(level2)
#         print(level3.shape)
#         # level 4
#         level4=self.level4(level3)
#         # residual connection
#         level4 = level4 + self.skip4(level3)
#         # upsample1
#         upsample1=self.upsample1(level4)
#         # concatenate level 3 and upsample1
#         concat1=torch.cat([upsample1, level3], dim=1)
#         # level 5
#         level5=self.level5(concat1)
#         # residual connection
#         level5 = level5 + self.skip5(concat1)
#         # upsample2
#         upsample2=self.upsample2(level5)
#         # concatenate level 2 and upsample2
#         concat2=torch.cat([upsample2, level2], dim=1)
#         # level 6
#         level6=self.level6(concat2)
#         # residual connection
#         level6 = level6 + self.skip6(concat2)
#         # upsample3
#         upsample3=self.upsample3(level6)
#         # concatenate level 1 and upsample3
#         concat3=torch.cat([upsample3, level1], dim=1)
#         # level 7
#         level7=self.level7(concat3)
#         # residual connection
#         level7 = level7 + self.skip7(concat3)
#         # final convolution
#         final_conv=self.final_conv(level7)
#         # sigmoid activation
#         sigmoid=self.sigmoid(final_conv)
#         return sigmoid
import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output