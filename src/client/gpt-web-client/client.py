import time
import pickle
from random import randrange
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains


class Client:
    def __init__(self) -> None:
        self.driver = uc.Chrome()
        self.load_cookies()
        self.get("https://chat.openai.com/chat")
        self.click_agreements()
        self.asked_count = 0

    def sleep(self):
        time.sleep(randrange(100, 250) / 100)

    def get(self, url: str):
        self.sleep()
        self.driver.get(url)

    def click(self, value: str):
        self.sleep()
        button = self.driver.find_element(By.XPATH, value)
        actions = ActionChains(self.driver)
        actions.move_to_element(button).perform()
        button.click()
    
    def send_keys(self, value: str, content: str):
        self.click(value)
        textarea = self.driver.find_element(By.XPATH, value)
        textarea.send_keys(content)
    
    def get_text(self, value: str) -> str:
        self.sleep()
        return self.driver.find_element(By.XPATH, value).text

    def close(self):
        self.sleep()
        cookies = self.driver.get_cookies()
        pickle.dump(cookies, open("cookies.pkl", "wb"))

    def load_cookies(self):
        self.sleep()
        self.get("https://chat.openai.com/404")
        cookies = pickle.load(open("cookies.pkl", "rb"))
        for cookie in cookies:
            try: self.driver.add_cookie(cookie)
            except Exception: continue

    def click_agreements(self):
        self.sleep()
        self.click("/html/body/div[3]/div/div/div/div[2]/div/div/div[2]/div[4]/button")
        self.click("/html/body/div[3]/div/div/div/div[2]/div/div/div[2]/div[4]/button[2]")
        self.click("/html/body/div[3]/div/div/div/div[2]/div/div/div[2]/div[4]/button[2]")
    
    def get_response(self):
        return self.get_text(f"/html/body/div[1]/div[2]/div[2]/main/div[1]/div/div/div/div[{2 * self.asked_count}]/div/div[2]")
    
    def wait_until_response(self) -> str:
        prev_answer, curr_answer = "true", "false"
        while prev_answer != curr_answer:
            prev_answer = self.get_response()
            time.sleep(5)
            curr_answer = self.get_response()
        return curr_answer

    def ask_question(self, content: str) -> str:
        self.sleep()
        self.send_keys("/html/body/div[1]/div[2]/div[2]/main/div[2]/form/div/div[2]/textarea", content)
        self.click("/html/body/div[1]/div[2]/div[2]/main/div[2]/form/div/div[2]/button")
        self.asked_count += 1
        return self.wait_until_response()


def main():
    client = Client()
    print(client.ask_question("How are you today?"))
    print(client.ask_question("What is your name?"))
    print(client.ask_question("What is your age?"))
    client.close()


if __name__ == "__main__":
    main()
