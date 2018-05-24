from setuptools import setup

setup(
    name='bcweather',
    version='0.0.1',
    description='Utility to incorporate future climate into weather files',
    author='James Hiebert',
    author_email='hiebert@uvic.ca',
    packages=['bcweather'],
    package_dir={'bcweather': 'bcweather'},
    package_data={'bcweather': ['tests/data/*']},
    include_package_data=True,
    zip_safe=False,
    scripts=['bcweather/gen_future_weather_file.py'],
    classifiers=[
        'Environment :: Console',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ]
)
