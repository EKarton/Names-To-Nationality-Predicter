"use strict";

let nameTextbox = document.getElementById('name-textbox');
nameTextbox.oninvalid = function(event) {
    if (!event.target.validity.valid) {
        event.target.setCustomValidity("Provide at least the first and last name");
    } else {
        event.target.setCustomValidity("");
    }
}
nameTextbox.oninput = function(event) {
    event.target.setCustomValidity("");
}
