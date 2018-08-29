# DRLTE
Project codes of ns3 and DRL

#create
cd DRLTE
git init
git add test.cc
git commit -m 'remark'
git remote add origin git@github.com:XiaoTianCan/DRLTE.git
git pull origin master#if there are updates remote
git push origin master

#update project(new file exists)
git add .
git commit
git push origin master

#update project(no new file, only delete or modify files)
git commit -a
git push origin master

#ignore some files
vim .gitignore #add file types into .gitignore, e.g. *.o, then git add . will ignore them*

#clone codes to local
git clone https://github.com/XiaoTianCan/DRLTE.git
#if there are codes locally and updates remotely, merge changes to local project
git fetch origin
git merge origin/master

#change branch
checkout branchname

#cancel
git reset

#delete
git rm *
