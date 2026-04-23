from setuptools import setup

package_name = "nav2_wfd"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sean Regan",
    maintainer_email="mail@mail.com",
    description="Wavefront frontier exploration node for ROS 2 Jazzy and Nav2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "explore = nav2_wfd.wavefront_frontier:main",
        ],
    },
)
