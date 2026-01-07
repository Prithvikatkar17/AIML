create database college ;

use college;

create table student (
	rollno int ,
    name varchar(30),
    age int
);

insert into student 
values
( 12 ,"bob",45),
(15,"ray",34);

select*from student;
