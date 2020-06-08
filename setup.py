from setuptools import setup, find_packages


dependency_links = []
with open('requirements.txt') as fp:
    install_requires = fp.read().split('\n')

if install_requires[-1] == "":
    install_requires = install_requires[:-1]

for i in range(len(install_requires)):
    if '#egg=' in install_requires[i]:
        pkg = install_requires[i].split('#egg=')[1]
        dependency_links.append(install_requires[i])
        install_requires[i] = pkg

print("Dependencies:\n{}".format(dependency_links))
print("Requirements:\n{}\n\n".format(install_requires))
PKG_NAME = "gbm_custom_loss"
print("Packages:")
packages = [PKG_NAME] + [PKG_NAME + '.' + pkg for pkg in find_packages(PKG_NAME)]
print(packages)

setup(name=PKG_NAME,
      version='0.0.2',
      description='GBM Custom Loss',
      author='Pavel Kochetkov',
      author_email='p02p@ya.ru',
      install_requires=install_requires,
      dependency_links=dependency_links,
      packages=packages,
      include_package_data=True)
