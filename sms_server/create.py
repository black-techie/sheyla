import smtplib





def send_email(plate):
    # sender = "Mobile phone Detection System <mailtrap@demomailtrap.com>"
    # receiver = "Authority <sheilahyl2001@gmail.com>"


    sender = "Private Person <mailtrap@demomailtrap.com>"
    receiver = "A Test User <willbroadnyirenda@gmail.com>"
    

    message = f"""\
    Urgent: Mobile Phone Usage Detected
    To: {receiver}
    From: {sender}

    This is to inform you that a car driver was detected using a mobile phone while driving. 
    \nNumber plate: {plate}"""

    with smtplib.SMTP("live.smtp.mailtrap.io", 587) as server:
        server.starttls()
        server.login("api", "9d7dc62657f496d56d1f7b7152eb806e")
        server.sendmail(sender, receiver, message)
    
    print("sent")

send_email(" T233 DEE")