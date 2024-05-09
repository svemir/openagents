<?php

declare(strict_types=1);

namespace App\AI;

use App\Models\Thread;
use GuzzleHttp\Client;
use Yethee\Tiktoken\EncoderProvider;

class SimpleInferencer
{
    // public until refactor
    public static float $messageTokens = 0;

    public static function inference(string $prompt, string $model, Thread $thread, callable $streamFunction, ?Client $httpClient = null): array
    {
        $modelDetails = Models::MODELS[$model] ?? null;

        if ($modelDetails) {
            $gateway = $modelDetails['gateway'];
            $maxTokens = $modelDetails['max_tokens'];

            $messages = [
                [
                    'role' => 'system',
                    'content' => 'You are a helpful assistant.',
                ],
                ...get_truncated_messages($thread, $maxTokens, $prompt),
            ];

            $completionTokens = $maxTokens - static::$messageTokens;

            if (! $httpClient) {
                $httpClient = new Client();
            }
            $params = [
                'model' => $model,
                'messages' => $messages,
                'max_tokens' => $completionTokens,
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
}

function get_truncated_messages(Thread $thread, int $maxTokens, string $prompt): array
{
    $provider = new EncoderProvider();
    $encoder = $provider->getForModel('gpt-4');

    $messages = [];
    $tokenCount = 0;
    $userContent = '';
    $maxTokensReached = false;

    // leave room for the actual prompt
    $promptTokens = count($encoder->encode($prompt));
    $maxTokens -= $promptTokens;

    foreach ($thread->messages()->orderBy('created_at', 'asc')->get() as $message) {

        // First check previously stored token count up to this point
        if ($message->input_tokens) {
            if ($message->input_tokens > $maxTokens) {
                $maxTokensReached = true;
                break;
            }
            // input_tokens contains the token count of the entire chat so far
            $tokenCount = $message->input_tokens;
        }

        if (is_null($message->model)) {
            if (strtolower(substr($message->body, 0, 11)) === 'data:image/') {
                $userContent .= ' <image>';
            } elseif (! str_contains($userContent, $message->body)) {
                $userContent .= ' '.$message->body;
            }
        } else {
            $userContent = trim($userContent);
            if (! empty($userContent)) {
                if (! $message->input_tokens) {
                    $messageTokens = count($encoder->encode($userContent));
                    if ($tokenCount + $messageTokens > $maxTokens) {
                        $maxTokensReached = true;
                        break;
                    }
                    $tokenCount += $messageTokens;
                }
                $messages[] = [
                    'role' => 'user',
                    'content' => $userContent,
                ];
                $userContent = '';
            }

            if (strtolower(substr($message->body, 0, 11)) === 'data:image/') {
                $content = '<image>';
            } else {
                $content = trim($message->body);
                if (empty($content)) {
                    // Some LLMs return a 400 error if they receive blank content
                    $content = '<blank>';
                }
            }

            $messageTokens = $message->output_tokens ?: count($encoder->encode($content));
            if ($tokenCount + $messageTokens > $maxTokens) {
                $maxTokensReached = true;
                break; // Stop adding messages if the remaining context is not enough
            }

            $messages[] = [
                'role' => 'assistant',
                'content' => $content,
            ];
            $tokenCount += $messageTokens;
        }
    }

    $lastRole = $messages ? end($messages)['role'] : '';

    if (! $maxTokensReached && ! empty($userContent)) {
        $messageTokens = count($encoder->encode($userContent));
        if ($tokenCount + $messageTokens > $maxTokens) {
            $maxTokensReached = true;
        } else {
            $lastRole = '';
        }
    }

    if ($maxTokensReached || ($lastRole === 'assistant')) {
        $userContent = $prompt;
        $messageTokens = $promptTokens;
    }

    if (! empty($userContent)) {
        $messages[] = [
            'role' => 'user',
            'content' => trim($userContent),
        ];
        $tokenCount += $messageTokens;
    }
    SimpleInferencer::$messageTokens = $tokenCount;

    return $messages;
}
