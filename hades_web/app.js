var express = require('express');
var path = require('path');
var favicon = require('serve-favicon');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');

var monk = require('monk');
var db = monk('localhost:27017/userlist');

var routes = require('./routes/index');
var showimgtag = require('./routes/showimgtag');

var app = express();

var session = require('express-session');
app.use(session({secret: 'mySecret', resave: false, saveUninitialized: false}));

// view engine setup --> jade
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

//view engine setup --> ejs
/*var engine = require('ejs-locals');
app.set('views', path.join(__dirname, 'views'));
app.engine('ejs', engine);
app.set('view engine', 'ejs');*/


app.use(logger('dev'));
//app.use(require('body-parser')());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use(function(req,res,next){
    req.db = db;
	next();
});

app.use('/', routes);
app.use('/', showimgtag);

/*
/// catch 404 and forwarding to error handler
app.use(function(req, res, next) {
    var err = new Error('Not Found');
    err.status = 404;
    next(err);
});

/// error handlers

// development error handler
// will print stacktrace
if (app.get('env') === 'development') {
    app.use(function(err, req, res, next) {
        res.status(err.status || 500);
        res.render('error', {
            message: err.message,
            error: err
        });
    });
}

// production error handler
// no stacktraces leaked to user
app.use(function(err, req, res, next) {
    res.status(err.status || 500);
    res.render('error', {
        message: err.message,
        error: {}
    });
});
*/



app.listen(3000, function () {
	console.log('Example app listening on port 3000!');
});


module.exports = app;
