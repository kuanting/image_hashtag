var express = require('express');
var router = express.Router();
var currentUser="";
var imageName="";
var imagePath="";
var success_info="";
//var imagePath = "./files/img001.jpg";


router.get("/", function(req, res) {
    res.render('references', { 
        
    });
});

module.exports = router;

