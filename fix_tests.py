
with open("tests/test_diffusion.py", "r") as f:
    content = f.read()

content = content.replace('''def test_diffusion_extra_coverage(mocker):
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)''', '''def test_diffusion_extra_coverage(mocker):
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)''')

with open("tests/test_diffusion.py", "w") as f:
    f.write(content)
