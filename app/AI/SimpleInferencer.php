<?php

declare(strict_types=1);

namespace App\AI;

use App\Models\Thread;
use GuzzleHttp\Client;
use Yethee\Tiktoken\EncoderProvider;

class SimpleInferencer
{
    private static int $remainingTokens = 0;

    private static $encoder;

    public static function inference(string $prompt, string $model, Thread $thread, callable $streamFunction, ?Client $httpClient = null): array
    {
        $modelDetails = Models::MODELS[$model] ?? null;

        if ($modelDetails) {
            $gateway = $modelDetails['gateway'];
            self::$remainingTokens = $modelDetails['max_tokens'];

            $messages = [
                [
                    'role' => 'system',
                    'content' => 'You are a helpful assistant.',
                ],
                ...self::getTruncatedMessages($thread),
            ];

            if (! $httpClient) {
                $httpClient = new Client();
            }
            $params = [
                'model' => $model,
                'messages' => $messages,
                'max_tokens' => self::$remainingTokens,
                'stream_function' => $streamFunction,
            ];
            switch ($gateway) {
                case 'meta':
                    $client = new TogetherAIGateway($httpClient);
                    break;
                case 'groq':
                    $client = new GroqAIGateway($httpClient);
                    break;
                case 'anthropic':
                    $client = new AnthropicAIGateway($httpClient);
                    break;
                case 'mistral':
                    $client = new MistralAIGateway($httpClient);
                    break;
                case 'openai':
                    $client = new OpenAIGateway();
                    break;
                case 'perplexity':
                    $client = new PerplexityAIGateway($httpClient);
                    break;
                case 'cohere':
                    $client = new CohereAIGateway($httpClient);
                    $params['message'] = $prompt;
                    break;
                case 'satoshi':
                    $client = new HuggingfaceAIGateway();
                    break;
                case 'greptile':
                    $client = new GreptileGateway();
                    $params = ['thread' => $thread];
                    break;
                default:
                    dd("Unknown gateway: $gateway");
            }
            $inference = $client->inference($params);
        } else {
            dd("Unknown model: $model");
        }

        return $inference;
    }

    public static function getTruncatedMessages(Thread $thread): array
    {
        $provider = new EncoderProvider();
        self::$encoder = $provider->getForModel('gpt-4');

        $messages = [];

        foreach ($thread->messages()->orderBy('created_at', 'desc')->get() as $message) {
            $role = is_null($message->model) ? 'user' : 'assistant';

            self::addMessage($role, $message, $messages);

            if (self::$remainingTokens < 0) {
                break;
            }
        }

        return $messages;
    }

    private static function addMessage(string $role, mixed $message, array &$messages): void
    {
        if (strtolower(substr($message->body, 0, 11)) === 'data:image/') {
            $content = '<image>';
        } else {
            $content = trim($message->body) ?: '<blank>';
        }
        $messageTokens = $message->output_tokens ?: count(self::$encoder->encode($content));

        if ($messageTokens <= self::$remainingTokens || ($role === 'user')) {
            self::$remainingTokens -= $messageTokens;

            if (count($messages) && $messages[0]['role'] === $role) {
                if (! str_contains($messages[0]['content'], $content)) {
                    $messages[0]['content'] = $messages[0]['content']."\n".$content;
                }
            } else {
                array_unshift($messages, [
                    'role' => $role,
                    'content' => $content,
                ]);
            }
        }
    }
}
