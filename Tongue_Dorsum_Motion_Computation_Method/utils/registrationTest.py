import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

path = './annotated/crx_mask'
# 加载源图像和目标图像
fixed_image = sitk.ReadImage(os.path.join(path, 'IMG-0001-00002.jpg'))
moving_image = sitk.ReadImage(os.path.join(path, 'IMG-0002-00002.jpg'))

# 将图像转换为浮点数类型
fixed_image = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkFloat32)
moving_image = sitk.Cast(sitk.RescaleIntensity(moving_image), sitk.sitkFloat32)
size = fixed_image.GetSize()
print(size)
#
# # 定义配准方法和参数
# registration_method = sitk.ImageRegistrationMethod()
# registration_method.SetMetricAsMattesMutualInformation()
# registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100)
# registration_method.SetInterpolator(sitk.sitkLinear)
#
# # 执行配准
# final_transform = registration_method.Execute(fixed_image, moving_image)
#
# # 应用变形场到源图像
# resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear)
#
# # 将源图像、目标图像和配准后的图像叠加显示
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 3, 1)
# plt.imshow(sitk.GetArrayViewFromImage(fixed_image), cmap='gray')
# plt.title('Fixed Image')
# plt.subplot(1, 3, 2)
# plt.imshow(sitk.GetArrayViewFromImage(moving_image), cmap='gray')
# plt.title('Moving Image')
# plt.subplot(1, 3, 3)
# plt.imshow(sitk.GetArrayViewFromImage(resampled_image), cmap='gray')
# plt.title('Registered Image')
# plt.tight_layout()
# plt.show()
#
# # 可视化变形场
# displacement_field = sitk.DisplacementFieldTransform(final_transform)
# displacement_field_arr = sitk.GetArrayFromImage(displacement_field)
# plt.imshow(displacement_field_arr[:, :, slice_index, 0], cmap='jet')
# plt.colorbar()
# plt.title('Displacement Field (X component)')
# plt.show()