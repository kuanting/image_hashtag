var express = require('express');
var router = express.Router();

var fileUpload = require('express-fileupload');
var cp = require('node-cp');
var async = require('async');
var sleep = require('sleep');

var showimgtag = require('./showimgtag');
router.use('/showimgtag', showimgtag);

var aboutsystem = require('./aboutsystem');
router.use('/aboutsystem', aboutsystem);

var references = require('./references');
router.use('/references', references);

var imagePath = "";
var debugFile = "";

const path = require('path')
const {spawn} = require('child_process')
function runScript(imageID){
  return spawn('python', [
    "-u", 
    path.join('./HashtagRec/Hashtag_end2end', 'main.py'),
    "-m", "test", "-id", imageID
  ]);
}

/* GET home page. */
router.get('/', function(req, res) {
    res.render('index', { 
		imagePath:imagePath, debugFile:debugFile
	});
	imagePath="";
	debugFile="";
});
router.use(fileUpload());

// get uploaded file
router.post("/", function(req, res){
   var d = new Date();
   start_time = d.getTime();
   var db = req.db;
   var sendFile;
   if(!req.files){
      debugFile = "you haven't sent any file";
      res.redirect("/");
   }  
   else if(req.files.sendFile.name.length == 0){
      console.log("no file actually!");
      debugFile = "you haven't sent any file";
      res.redirect("/");
   } 
   else{
      sendFile = req.files.sendFile;
      var dt = new Date();
      imageID = dt.getTime()+".jpg";
      imagePath = './files/'+imageID;
      sendFile.mv('./public/files/'+imageID, function(err){
         if(err){
            debugFile = err;
            res.status(500).redirect("/");
         }
         else{
                //debugFile = "Upload successfully and Wait for the result...";
                console.log(imageID);
                const subprocess = runScript(imageID);
                subprocess.stdout.on('data', (data) => {
                  console.log(`data:${data}`);
                });
                subprocess.stderr.on('data', (data) => {
                  console.log(`error:${data}`);
                });
                subprocess.stderr.on('close', () => {
                  console.log("Closed");
                });
                var session = req.session;
                session.message = imageID;
                now_time = d.getTime();
                //console.log(now_time - start_time);
                sleep.sleep(6);
                res.redirect("/showimgtag");
                
         }
    });
   }
});


module.exports = router;


