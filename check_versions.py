import pkg_resources

# 설치된 모든 패키지 가져오기
installed_packages = pkg_resources.working_set

# 패키지 이름과 버전을 정렬하여 출력
for package in sorted(installed_packages, key=lambda x: x.key):
    print(f"{package.key}=={package.version}") 