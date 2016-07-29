# NICE (Northeastern Interactive Clustering Engine)
The Northeastern Interactive Clustering Engine (NICE) is an open source 
data analysis framework, which aims to helping researchers in different 
domains gain insight in their data by providing an interactive 
webpage-based interface with a set of clustering algorithms and the 
capability to visualize the clustering results. The framework is still 
under development.

## For Contributors:
There are two ways to contribute to this project. If you are added to the project as a collaborator, please follow the steps in "Using Branch" section. Otherwise, you can fork the project and submit pull requests; the instructions are listed in "Using Fork" section. The most important rule here is that we only use pull request to contribute and we never push directy to the master or develop branch.

### Using Branch:
1. Clone the repository: `git clone git@github.com:yiskylee/NICE.git`.
2. Create your own local feature branch: `git checkout -b your-own-feature-branch develop`
3. Make your own feature branch visible by pushing it to the remote repo (DO NOT PUSH IT TO THE DEVELOP BRANCH): `git push --set-upstream origin your-own-feature-branch`
4. Develop your own feature branch in your local repository: `git add`, `git commit`, etc..
5. After your own branch is completed, make sure to merge the latest change from the remote develop branch to your own local develop branch: 1) `git checkout develop` 2) `git pull`.
6. Now that your local develop branch is up to date, you can update your own feature branch by: 1) `git checkout your-own-feature-branch` 2) `git pull origin develop`.
7. Update your own feature branch on the remote repository by: `git push origin your-own-feature-branch`
8. Make a pull request with base being develop and compare being your-own-feature-branch
9. After the pull request is merged, your-own-feature-branch on the remote repository will be soon deleted, delete it on your local repository by: `git branch -d your-own-feature-branch`

### Using Fork:
1. Fork the repository to your own remote repository.
2. Git clone the repository: `git clone git@github.com/your_account_name/NICE.git`
3. Add this project as an upstream to your local repository by `git remote add upstream https://github.com/yiskylee/NICE.git`. You can use `git remote -v` to view the upstream.
4. Create your own local feature branch: `git checkout -b your-own-feature-branch develop`
3. Make your own feature branch visible by pushing it to your own remote repository (DO NOT PUSH IT TO THE DEVELOP BRANCH): `git push --set-upstream origin your-own-feature-branch`
4. Develop your own feature branch in your local repository: `git add`, `git commit`, etc..
5. After your own branch is completed, make sure to merge the latest change from upstream develop branch to your own origin develop branch: 1) `git checkout develop` 2) `git pull upstream develop` 3) `git push origin develop`
6. Since that you have the latest change in your own origin develop branch from upstream one, now you can update your own feature branch on the your own remote repository by: 1) `git checkout your-own-feature-branch` 2) `git pull origin develop` 3) `git push origin your-own-feature-branch`
7. Make a pull request from your own feature branch on your own remote repository on github to the develop branch of this repository.
8. After the pull request is merged, you can delete your own feature branch by 1) `git push origin --delete your-own-feature-branch` to delete the remote branch and 2) `git branch -d your-own-feature-branch` to delete your local branch.
9. More instructions on using fork can be found [here](https://help.github.com/articles/fork-a-repo/).

## Compile and Test Nice:
We use CMake tool to automatically build and test the framework. After you download the repository, you need to go to NICE/cpp and run `./configure.sh`. This is only a one time operation as it will create a build directory where all executables generated will be put into. To build the code and the tests, go to build directory and run 1) `make` 2) `make test ARGS="-V"`.

## Coding Style:
We are following [Google c++ style guide](https://google.github.io/styleguide/cppguide.html), make sure to use `google_styleguide/cpplint/cpplint.py` to check your code and make sure there are no errors. Additionally, `cpplint.py` has been integrated to Nice together with cmake, so you should be able to check your code through cmake-generated Makefile. After you run `./configure.sh` indicated in previous section, go to build directory and run `make check`.
For developers preferring IDE like Eclipse, you can also import `eclipse-cpp-google-style.xml`(Can be found from [Google c++ style guide](https://google.github.io/styleguide/cppguide.html)) into Eclipse to auto-format your code before using `cpplint.py` or `make check`.

## Documentation
We are using Doxygen to automatically generate project documents. To produce the html based documents, you should run `make doc` after you run `./configure.sh`. Make sure Doxygen is intalled on your computer. For Ubuntu users, type command `sudo apt-get install doxygen` to intall it. For more information about Doxygen, check their [official website](http://www.stack.nl/~dimitri/doxygen/).
All documents will be generated under directory doc/html. Double click index.html to browse generated documents from any web browser you like(Chrome, Firefox etc.)

## Hosting documentation on Github pages
We will be using Github pages in order to publically host our compiled doxygen files.

Considering that we plan to support program language other than C++ for NICE, we decide to host documentation of different program language in separate repositories. For C++ version, the doxygen-genreated documentation is located in [here](https://yiskylee.github.io/NiceCppDoc/)

Due to the complexity of working with submodule in git, the documentation maintainance and publishing will only be done by contributors using branch. Therefore, contributors using fork should only be responsible of documenting through doxygen comments in source code. Besides, they can generate local documents by following normal procedure described as previous section. 

### Steps for creating and publishing NICE documentation
We are using submodule tools of git to assoicate documentation repository with NICE repository

1. Clone a documentation repository
`
$ git clone <linkToCloneDocRepo>
`
2. Create a gh-pages branch (gh-pages is a special branch in github that aims to hosting html-based documentation directly on-line)
`
$ git checkout --orphan gh-pages
$ rm -rf *
$ git add .
$ git commit -m "Initialize gh-pages branch as empty directory"
$ git push origin gh-pages
`
3. Go back to NICE repository
4. Set up a submodule
`
$ git submodule add -b gh-pages <linkToCloneDocRepo> cpp/doc
`
5. Generate the documentation through doxygen and push it to gh-pages branch

### Steps for updating NICE documentation
(Assume that document submodule have been established)

1. Clone a NICE repo
2. Initialize the submodule
`
$ git submodule update --init
`
3. Go inside doc directory and checkout gh-pages branch(This step has to be done because the a dettached HEAD will be returned for the submodule)
4. Generate the documentation through doxygen and push it to gh-pages branch
