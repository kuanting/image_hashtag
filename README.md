# image_hashtag
Image Hashtag Recommendation System

## HADES website
### Structure of Web server
- Front-end: Jade + Javascript
- Back-end: Node.js + Express + npm
- Database: Mongodb

### Installation
- nodejs and npm
> Ref: http://tecadmin.net/install-latest-nodejs-npm-on-ubuntu/
- Express
> Ref: http://expressjs.com/zh-tw/starter/installing.html
- MongoDB
> Ref: https://docs.mongodb.com/manual/tutorial/install-mongodb-enterprise-on-ubuntu/

### Usage
- For linux 
```c
$ ./Build.sh
```

- Things in <strong>Build.sh</strong>:
```c
$ sudo service mongod start
$ npm i
$ npm start //or node app.js
```

- go to <strong>localhost:3000/</strong> in your browser
