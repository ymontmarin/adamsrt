from setuptools import setup, find_packages

# Find requirements
try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements
requirements_parsing = list(parse_requirements('./requirements.txt',
                                               session=False))
install_requires = [str(r.req) for r in requirements_parsing]

setup(
    name="adamsrt",
    version='0.1.0',
    author="Yann de Mont-Marin, Simon Roburin",
    author_email='yann.montmarin@gmail.com',
    url="https://github.com/ymontmarin/adamsrt",
    description="New optimization schemes AdamSRT and AdamS"
                " from 'Spherical Perspective on Learning with Batch Norm'",
    packages=find_packages(),
    zip_safe=False,
    python_requires='>=3',
    install_requires=install_requires,
)
