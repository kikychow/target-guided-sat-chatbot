import React from "react";

const MessageParser = ({ children, actions }) => {
  const getResponse = async (message) => {
    actions.handleHello(message)

    // let response = await fetch("http://127.0.0.1:5000/predict", {
    //   method: "POST",
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ message: message })
    // });

    // if (response.ok) {
    //   var json_data = await response.json();
    //   if (json_data.success) {
    //     actions.handleMessage(json_data["answer"]);
    //   }
    // }
  };

  // const test = async (message) => {
  //   let response = await fetch("/hello")
  //   if (response.ok) {
  //     var json_data = await response.json();
  //     console.log(json_data);
  //     if (json_data.success) {
  //       actions.handleMessage("hi");
  //     }
  //   }
  // };

  

  const parse = (message) => {
    // if (message.includes("hello")) {
    //   actions.handleHello();
    // }
    getResponse(message);
    // test(message);
  };

  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          parse: parse,
          actions: {},
        });
      })}
    </div>
  );
};

export default MessageParser;
