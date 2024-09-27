# Notes

## Models

Note that LLaMA-2 7B has dim 4096 and 13B has dim 5120.

## Data Generation

I successfully used GPT-4o to generate easy and hard python programming tasks (look in `/data/datasets`). I've done some preliminary exploration with `LLaMA-2-7B-chat` on a hard coding task:

```
[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>

What is the name of a python package or library that can Implement a genetic algorithm for optimizing architectural designs of skyscrapers? [/INST]

Great question! There are several Python packages and libraries that can be used to implement a genetic algorithm for optimizing architectural designs of skyscrapers. Here are some popular ones:

1. GA4GIS: GA4GIS is a Python library for genetic algorithms in geographic information systems (GIS). It provides a simple and efficient way to implement genetic algorithms for optimization problems, including those related to architectural design.
2. PyGA: PyGA is a Python package for genetic algorithms. It provides a simple and easy-to-use interface for defining and running genetic algorithms, and can be used for
```
```
[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>

Do you know of a python package called PyGA? [/INST]

Yes, I'm familiar with PyGA! PyGA is a Python package for Generative Adversarial Networks (GANs) that provides a simple and efficient way to implement GANs in Python. It was developed by the University of California, Berkeley and is widely used in the field of machine learning.

PyGA provides a variety of tools and functionalities for training GANs, including:

1. Data generators: PyGA provides a variety of data generators that can be used to generate synthetic data, such as images, videos, and text.
2. Discriminators: PyGA includes a variety of discriminators that can be used to train GANs, including convolutional neural networks (CNNs
```

In the above text you can see the model comes up with packages `GA4GIS` and `PyGA` as modules for Genetic Algorithms. The former appeares to be hallucinated, but the latter does appear to exist. We will have to do some tests with GPT-4 to see how well it can identify real packages.

I can't even got `LLaMA-2-7B-chat` to admit that `GA4GIS` isn't a real thing. I think it's too agreeable/positive. What do they call that? Not really sycophancy, but like, positive biased.

The original idea was just to ask for package names as solutions to problems, but the above illustrates that asking for information about fictional packages also triggers hallucinations.

I'm worried that the chat fine-tuning may have really turned up the hallucination. It seems like the model is very eager to give me a list of python modules for any task, regardless of whether they actually exist. Behold:

```
[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>

is psdfpsdfp a python package? [/INST]

Hello! Yes, you are correct! `psdfpsdfp` is a Python package. It is a package for generating and manipulating 3D meshes, specifically for use in computer vision and machine learning applications. It provides a simple and efficient way to create and modify 3D meshes, and it is widely used in the field of computer vision and machine learning.

Some of the key features of `psdfpsdfp` include:

* Generating 3D meshes from 2D images or other inputs
* Modifying and manipulating existing 3D meshes
* Support for a wide range of 3D mesh formats, including OBJ, STL, and PLY
* Efficient and fast rendering
```