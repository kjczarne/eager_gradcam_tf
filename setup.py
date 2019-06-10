import setuptools
with open("README.md", "r") as f:
    long_description = f.read()
setuptools.setup(
     name='eagergradcamtf',  
     version='0.1',
     author="Krzysztof J. Czarnecki",
     author_email="kjczarne@gmail.com",
     description="",
     py_modules=['eagergradcamtf'],
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