#!/usr/bin/env python3


def main():
    process_repos()


def process_repos():
    with open("datasets/raw/random_repos.txt", "r") as fileobj:
        repos = [line.rstrip("\n") for line in fileobj]

    repos_list = [
        repo.replace("https://github.com/", "") for repo in repos if repo != "null"
    ]

    content = "\n".join(repos_list)
    print(content, end="")

    with open("datasets/raw/random_parsed.txt", "w") as file:
        file.write(content)


if __name__ == "__main__":
    main()
