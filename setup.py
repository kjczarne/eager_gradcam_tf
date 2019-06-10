import setuptools
with open("README.md", "r") as f:
    long_description = f.read()
setuptools.setup(
     name='eager_gradcam_tf',  
     version='0.1',
     scripts=['eager_gradcam.py'] ,
     author="Krzysztof J. Czarnecki",
     author_email="kjczarne@gmail.com",
     description="",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/kjczarne/eager_gradcam_tf",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU GPL-3.0 License",
         "Operating System :: OS Independent",
     ],
 )