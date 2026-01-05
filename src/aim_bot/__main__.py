# aim/app/bot/__main__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

# This is an experimental bot that is not used in the main application, but has been used in the past.

import logging
import hikari
import lightbulb
import time

from aim.config import ChatConfig
from aim.chat.manager import ChatManager, ROLE_USER, ROLE_ASSISTANT
from aim.chat.strategy import ChatTurnStrategy, chat_strategy_for

logger = logging.getLogger(__name__)


class ChatBot:
    """
    A Discord chat bot implemented using hikari and lightbulb frameworks.
    Features:
    - Responds to /ping command.
    - Listens to messages in a specified channel and forwards them to a handler.
    - Handles guild join events by sending a welcome message.
    """

    def __init__(self, chat: ChatManager, chat_strategy: ChatTurnStrategy, discord_app_id: str, discord_bot_token: str, discord_public_key: str, config: ChatConfig):

        self.chat = chat
        self.persona = chat.roster.personas[config.persona_id]
        self.chat_strategy = chat_strategy
        self.discord_app_id = discord_app_id
        self.discord_bot_token = discord_bot_token
        self.discord_public_key = discord_public_key
        self.chat_config = config
        self.last_response_time = 0

        if not self.discord_app_id:
            logger.error("Bot token not found. Please set DISCORD_APP_ID in environment variables.")
            raise ValueError("Api ID not provided.")

        if not self.discord_bot_token:
            logger.error("Bot token not found. Please set DISCORD_BOT_TOKEN in environment variables.")
            raise ValueError("Bot token not provided or invalid.")

        if not self.discord_public_key:
            logger.error("Public Key not found. Please set DISCORD_PUBLIC_KEY in environment variables.")
            raise ValueError("Public Key not provided or invalid.")

        # Initialize hikari GatewayBot
        self.bot = hikari.GatewayBot(
            token=self.discord_bot_token,
            intents=hikari.Intents.ALL_UNPRIVILEGED | hikari.Intents.MESSAGE_CONTENT
        )

        # Initialize lightbulb client from hikari bot
        self.client = lightbulb.client_from_app(self.bot)

        # Subscribe lightbulb to bot events
        self.bot.subscribe(hikari.StartingEvent, self.client.start)
        self.bot.subscribe(hikari.StoppingEvent, self.client.stop)

        # Register commands and event listeners
        self.register_commands()
        self.register_event_listeners()

    def register_commands(self):
        """Registers bot commands."""
        @self.client.register()
        class Ping(
            # Command type - builtins include SlashCommand, UserCommand, and MessageCommand
            lightbulb.SlashCommand,
            # Command declaration parameters
            name="ping",
            description="checks the bot is alive",
        ):
            # Define the command's invocation method. This method must take the context as the first
            # argument (excluding self) which contains information about the command invocation.
            @lightbulb.invoke
            async def invoke(self_i, ctx: lightbulb.Context) -> None:
                # Send a message to the channel the command was used in
                latency = self.bot.heartbeat_latency * 1000  # Convert to milliseconds
                await ctx.respond(f'🏓 Pong! Latency: {latency:.2f} ms')

        logger.info("Commands registered.")

    def register_event_listeners(self):
        """Registers event listeners for the bot."""

        @self.bot.listen(hikari.GuildJoinEvent)
        async def on_guild_join(event: hikari.GuildJoinEvent) -> None:
            guild_name = event.guild.name
            guild_id = event.guild.id
            logger.info(f"Joined guild: {guild_name} (ID: {guild_id})")

            # Attempt to send a welcome message to the system channel
            system_channel_id = event.guild.system_channel_id
            if system_channel_id:
                try:
                    await self.bot.rest.create_message(
                        system_channel_id,
                        f"Hello, {guild_name}! Thanks for inviting me to your server. Use `/ping` to check if I'm online."
                    )
                    logger.info(f"Sent welcome message to guild: {guild_name}")
                except hikari.ForbiddenError:
                    logger.warning(f"Insufficient permissions to send messages in guild: {guild_name}")

        @self.bot.listen(hikari.MessageCreateEvent)
        async def on_message(event: hikari.MessageCreateEvent) -> None:
            # Ignore messages from bots
            if event.is_bot:
                return

            # Check if the message is in the specified channel
            await self.handle_message(event.channel_id, event.message)

        logger.info("Event listeners registered.")

    async def handle_message(self, channel_id: str, message: hikari.Message):
        """
        Handles messages received in the specified channel.
        You can expand this method to include more complex logic.
        """
        author = message.author
        content = message.content

        logger.info(f"Message received in channel {channel_id} from {author}: {content}")
        # We will track the last time we responded. We will just add the message to the history
        # if the time is within the last 30 seconds.
        user_id = author.username
        # TODO set the user id
        self.chat_config.system_message = self.persona.system_prompt(location="In the discord chatroom.", user_id=user_id)
        curtime = time.time()
        if curtime - self.last_response_time < 30:
            self.chat.add_history(role=ROLE_USER, content=content, author=user_id)
            logger.info("Not responding to message because it is too soon.")
            return

        # Example: Echo the message back to the channel
        chat_turns = self.chat_strategy.chat_turns_for(self.persona, content, self.chat.history)
        # TODO add a provider
        try:
            chunks = []
            for t in self.chat.llm.stream_turns(chat_turns, self.chat.config):
                if t is not None:
                    print(t, end='', flush=True)
                    chunks.append(t)
                else:
                    print('', flush=True)

            response = ''.join(chunks)

            await message.respond(response)
            self.chat.add_history(role=ROLE_USER, content=content, author=user_id)
            self.chat.add_history(role=ROLE_ASSISTANT, content=content, author=self.chat_config.persona_id)
            logger.info("Echoed message back to the channel.")
        except hikari.ForbiddenError:
            logger.error("Failed to send message. Missing permissions.")
        except Exception as e:
            logger.exception(f"An error occurred while handling the message: {e}")

    def run(self):
        """Starts the bot."""
        logger.info("Starting the bot...")
        self.bot.run()

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'ChatBot':
        """
        Creates a ChatBot instance from a ChatConfig.
        """
        chat = ChatManager.from_config(config)
        chat_strategy = chat_strategy_for(chat.persona.chat_strategy, chat)
        return cls(chat=chat, chat_strategy=chat_strategy, discord_app_id=config.discord_app_id, discord_bot_token=config.discord_bot_token, discord_public_key=config.discord_public_key, config=config)

if __name__ == "__main__":
    try:
# Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        config = ChatConfig.from_env()
        
        chatbot = ChatBot.from_config(config=config)
        chatbot.run()
    except Exception as e:
        logger.exception(f"Failed to start the bot: {e}")
