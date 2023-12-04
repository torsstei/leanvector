python setup.py register sdist

VERSION=$(sed -n 's/^Version: *\([^ ]*\)/\1/p' leanvector.egg-info/PKG-INFO)
if [[ -z "${VERSION}" ]]; then echo 'Error: No package version found.'; exit 1; fi
TAG="pip-${VERSION}"
echo "Tagging current branch as $TAG"
git tag -a ${TAG} -m "Release ${VERSION} of leanvector pip package"
git push origin ${TAG}

echo "Publishing leanvector version ${VERSION}"
twine upload --skip-existing dist/*