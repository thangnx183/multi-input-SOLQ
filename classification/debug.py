import timm
model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k', pretrained=True,num_classes=2)

print(model)