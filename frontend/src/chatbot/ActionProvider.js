import React from "react";
import Loader from "./Loader.tsx";

const ActionProvider = ({ createChatBotMessage, setState, children }) => {
  const handleMessage = (botResponse) => {
    console.log(botResponse);
    const botMessage = createChatBotMessage(botResponse);

    setState(
      (prev) => (
        console.log(prev),
        {
          ...prev,
          messages: [...prev.messages.slice(0, -1), botMessage],
        }
      )
    );
  };

  const handleHello = async (user_input) => {
    let botMessage = createChatBotMessage('', {
      loading: true,
    });

    setState(
      (prev) => (
        console.log(prev),
        {
          ...prev,
          messages: [...prev.messages, botMessage],
        }
      )
    );

    let response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: user_input })
    });

    if (response.ok) {
      var json_data = await response.json();
      if (json_data.success) {
        // console.log(botMessage["message"])
        // botMessage["message"] = json_data["answer"];
        setState(
          (prev) => (
            console.log(prev),
            
            {
              ...prev,
              message: json_data["answer"],
            }
          )
        );
      }
    }

    // setState(
    //   (prev) => (
    //     console.log(prev),
    //     {
    //       ...prev,
    //       messages: [...prev.messages, botMessage],
    //     }
    //   )
    // );
    // setState((prev) => (console.log(prev),{
    //   ...prev,
    //   messages: [...prev.messages.slice(0, -1), {...prev.messages[-1]}],
    // }));
    // setState((prev) => ({ ...prev, messages: [...prev.messages[-1]] }));
  };
  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          actions: { handleMessage, handleHello },
        });
      })}
    </div>
  );
};

export default ActionProvider;
