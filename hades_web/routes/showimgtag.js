var express = require('express');
var router = express.Router();
var currentUser="";
var imageName="";
var imagePath="";
var success_info="";
//var imagePath = "./files/img001.jpg";


router.get("/", function(req, res) {
    currentUser = req.query.currentUser;

    var imageName = req.session.message;

    var monk = require('monk');
    var db = monk('localhost:27017/taglist');
    var collection = db.get('imageTag');

    collection.find({imgID:imageName}, function(e,docs){  
        res.render('showimgtag', {
            tags: docs,
            currentUser: currentUser,
            imagePath: "./files/"+imageName, 
	    success_info: success_info
        }); 
        imagePath="";
        currentUser="";
	success_info="";
    });
});


router.post("/", function(req, res){
	/* Something store user feedback tinto mongoDB */
    console.log("ddddddddd");
    success_info = "Feedback submit sucessfully! Thank you very much!";


});


module.exports = router;

