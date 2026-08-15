
with open("src/models/diffusion/__init__.py", "r") as f:
    content = f.read()

content = content.replace("from .configuration_diffusion import DiffusionConfig", "__all__ = ['DiffusionConfig', 'UniversalDiffusionLM', 'DiffusionModelForConditionalGeneration']\nfrom .configuration_diffusion import DiffusionConfig")

with open("src/models/diffusion/__init__.py", "w") as f:
    f.write(content)
