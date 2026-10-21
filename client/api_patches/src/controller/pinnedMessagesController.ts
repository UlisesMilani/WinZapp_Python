import type { Request, Response } from 'express';
import type { Message } from '@wppconnect-team/wppconnect';
import { readPinnedMessageIds } from '../util/pinnedMessagesRuntime';

export async function readPinnedMessages(req: Request, res: Response) {
  const { chatIds } = req.body || {};
  if (!Array.isArray(chatIds) || chatIds.length < 1 || chatIds.length > 3
      || !chatIds.every(id => typeof id === 'string'
        && /^\d+(-\d+)?@(c\.us|s\.whatsapp\.net|lid|g\.us)$/.test(id))) {
    return res.status(400).json({ status: 'error', code: 'invalid_chat_ids' });
  }
  const page = (req.client as any)?.page;
  if (!page || page.isClosed()) {
    return res.status(503).json({ status: 'error', code: 'pinned_messages_unavailable' });
  }
  try {
    const ids = await page.evaluate(readPinnedMessageIds, chatIds);
    const messages: (Message & { pinInChat: boolean })[] = [];
    // Use WPPConnect's serializer, the same shape as get-messages, not MsgModel.toJSON().
    for (const id of ids) {
      const message = await req.client.getMessageById(id);
      if (!message?.id) throw new Error('pinned_messages_invalid');
      messages.push({ ...message, pinInChat: true });
    }
    return res.status(200).json({ status: 'success', response: messages });
  } catch (_) {
    // Native errors can include phone numbers and message content.
    return res.status(503).json({ status: 'error', code: 'pinned_messages_unavailable' });
  }
}
