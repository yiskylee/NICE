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
1. Git clone the repository: `git clone git@github.com:yiskylee/NICE.git`
2. Create your own local feature branch: `git checkout -b your-own-feature-branch develop`
3. Make your own feature branch visible by pushing it to the remote repo (DO NOT PUSH IT TO THE DEVELOP BRANCH): `git push origin your-own-feature-branch`
4. Develop your own feature branch in your local repository: `git add`, `git commit`, etc..
5. After your own branch is completed, make sure to merge the latest development branch to your own feature branch: 1) `git checkout your-own-feature-branch` 2) `git pull origin develop`
6. Update your own feature branch on the remote repository by: `git push origin your-own-feature-branch`
7. Make a pull request with base being develop and compare being your-own-feature-branch
8. After the pull request is merged, your-own-feature-branch on the remote repository will be soon deleted, delete it on your local repository by: `git branch -d your-own-feature-branch`

### Using Fork:
1. Fork the repository to your own remote repository.
2. Git clone the repository: `git clone git@github.com:your_account_name/NICE.git`
3. Add this project as an upstream to your local repository by `git remote add upstream https://github.com/yiskylee/NICE.git`. You can use `git remote -v` to view the updatream.
3. Make your own feature branch visible by pushing it to your own remote repository (DO NOT PUSH IT TO THE DEVELOP BRANCH): `git push origin your-own-feature-branch`
4. Develop your own feature branch in your local repository: `git add`, `git commit`, etc..
5. After your own branch is completed, make sure to merge the latest development branch to your own featrue branch: 1) `git checkout your-own-feature-branch` 2) `git pull upstream develop`
6. Update your own feature branch on the your own remote repository by: `git push origin your-own-feature-branch`
7. You should also update the develop branch on your own remote repository by: 1) `git checkout develop` 2) `git pull upstream develop` 3) `git push origin develop`
8. Make a pull request from your own feature branch on your own remote repository on github to the develop branch of this repository.
9. After the pull request is merged, you can delete your own feature branch by 1) `git push origin --delete your-own-feature-branch` to delete the remote branch and 2) `git branch -d your-own-feature-branch` to delete your local branch.

## Coding Style:
We are following [Google c++ style guide](https://google.github.io/styleguide/cppguide.html), make sure to use `google_styleguide/cpplint/cpplint.py` to check your code and make sure there are no errors. You can also import `google_styleguide/eclipse-cpp-google-style.xml` into Eclipse to auto-format your code before using `cpplint.py`.
