var express = require('express');
var router = express.Router();
var currentUser="";
var imageName="";
var imagePath="";
var success_info="";

router.get("/", function(req, res) {
    imageName = req.session.message;

    var monk = require('monk');
    var db = monk('localhost:27017/taglist');
    var collection = db.get('imageTag');

    collection.find({imgID:imageName}, function(e,docs){  
        res.render('showimgtag', {
            tags: docs,
            imagePath: "./files/"+imageName, 
	    success_info: success_info
        }); 
        imagePath="";
        success_info="";
    });
});

// get uploaded file
router.post("/", function(req, res){
   //console.log("cccccccccc");
   var d = new Date();
   start_time = d.getTime();
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
                res.redirect("/");
                
         }
    });
   }
});

router.post("/showimgtag/feedback", function(req, res){
    /* Something store user feedback tinto mongoDB */
    var monk = require('monk');
    var db = monk('localhost:27017/taglist');
    var collection = db.get('imageTag');
    var f1="-1", f2="-1", f3="-1", f4="-1", f5="-1", f6="-1", f7="-1", f8="-1", f9="-1", f10="-1";
    if(req.body.tag1 == 'on')
        f1 = "1";
    if(req.body.tag2 == 'on')
        f2 = "1";
    if(req.body.tag3 == 'on')
        f3 = "1";
    if(req.body.tag4 == 'on')
        f4 = "1";
    if(req.body.tag5 == 'on')
        f5 = "1";
    if(req.body.tag6 == 'on')
        f6 = "1";
    if(req.body.tag7 == 'on')
        f7 = "1";
    if(req.body.tag8 == 'on')
        f8 = "1";
    if(req.body.tag9 == 'on')
        f9 = "1";
    if(req.body.tag10 == 'on')
        f10 = "1";

    collection.update({imgID:imageName}, { $set: { one_f:f1, two_f:f2, three_f:f3, four_f:f4, five_f:f5, six_f:f6, seven_f:f7, eight_f:f8, nine_f:f9, ten_f:f10} }, {w:1}, function(e,result){ 
        if(e) throw e; 
        console.log("MongoDB update successfully.");
        res.redirect("/showimgtag");
    });

});


module.exports = router;

